from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

class PrivilegeLevel:
    READ = 'read'
    WRITE = 'write'
    EXECUTE = 'execute'

class EffectType:
    NO_SIDE_EFFECT = 'no_side_effect'
    STATE_MUTATION = 'state_mutation'
    EXTERNAL_COMMUNICATION = 'external_communication'
    SHARING = 'sharing'
    DELETION = 'deletion'
    FINANCIAL = 'financial'
    CODE_EXECUTION = 'code_execution'
    SECURITY_ACCOUNT = 'security_account'
    COMMITMENT = 'commitment'
    UNKNOWN_HIGH_RISK = 'unknown_high_risk'

@dataclass
class GateDecision:
    tool_name: str
    privilege: str
    action: str
    reason: str = ''
    effect_type: str = EffectType.UNKNOWN_HIGH_RISK
    arg_roles: list[str] = field(default_factory=list)
    destination_values: list[str] = field(default_factory=list)
    selector_values: list[str] = field(default_factory=list)
    temporal_values: list[str] = field(default_factory=list)
    target_values: list[str] = field(default_factory=list)
    payload_values: list[str] = field(default_factory=list)
    sensitive_values: list[str] = field(default_factory=list)
    is_external_sink: bool = False
    is_high_consequence: bool = False
_READ_VERBS = {'search', 'query', 'get', 'read', 'list', 'fetch', 'find', 'lookup', 'check', 'inspect', 'view', 'show', 'open', 'retrieve'}
_WRITE_VERBS = {'write', 'edit', 'update', 'modify', 'set', 'create', 'add', 'append', 'insert', 'replace', 'revise', 'save', 'store', 'record'}
_DELETE_VERBS = {'delete', 'remove', 'erase', 'wipe', 'drop', 'destroy'}
_EXTERNAL_VERBS = {'send', 'post', 'submit', 'upload', 'share', 'publish', 'notify', 'invite', 'message', 'email', 'forward', 'deliver'}
_SHARE_VERBS = {'share', 'invite', 'publish', 'expose', 'grant', 'attach'}
_FINANCIAL_VERBS = {'pay', 'transfer', 'withdraw', 'deposit', 'purchase', 'refund', 'charge'}
_EXEC_VERBS = {'run', 'exec', 'execute', 'deploy', 'install', 'launch', 'start'}
_SECURITY_ACCOUNT_VERBS = {'login', 'logout', 'password', 'credential', 'auth', 'token', 'verify', 'reset'}
_COMMIT_VERBS = {'book', 'schedule', 'reserve', 'confirm', 'register', 'enroll'}
_COMMITMENT_WRITE_VERBS = {'create', 'add', 'update', 'set', 'save', 'record', 'modify'}
_COMMITMENT_OBJECT_NOUNS = {'calendar', 'event', 'events', 'meeting', 'meetings', 'appointment', 'appointments', 'reservation', 'reservations', 'booking', 'bookings', 'itinerary', 'itineraries', 'reminder', 'reminders', 'invite', 'invitation', 'invitations', 'hotel', 'hotels', 'restaurant', 'restaurants', 'flight', 'flights', 'car', 'cars', 'trip', 'trips', 'table', 'tables', 'venue', 'venues'}
_DESTINATION_KEYS = {'recipient', 'recipients', 'email', 'emails', 'phone', 'phones', 'address', 'addresses', 'url', 'urls', 'endpoint', 'endpoints', 'webhook', 'callback', 'callback_url', 'destination', 'destinations', 'target', 'targets', 'to', 'cc', 'bcc'}
_INTERNAL_TARGET_KEYS = {'participant', 'participants', 'attendee', 'attendees', 'invitee', 'invitees', 'contact', 'contacts', 'channel', 'channels', 'member', 'members', 'user', 'users'}
_SELECTOR_KEYS = {'name', 'names', 'resource', 'resources', 'entity', 'entities', 'item', 'items', 'account', 'accounts', 'file', 'files', 'document', 'documents', 'venue', 'venues', 'property', 'properties', 'company', 'companies', 'hotel', 'restaurant'}
_TEMPORAL_KEYS = {'date', 'dates', 'time', 'times', 'start', 'end', 'day', 'days', 'start_date', 'end_date', 'start_day', 'end_day', 'start_time', 'end_time'}
_PAYLOAD_KEYS = {'content', 'message', 'body', 'description', 'text', 'payload', 'note', 'comment', 'summary', 'subject', 'title', 'attachment', 'attachments'}
_SENSITIVE_KEYS = {'password', 'secret', 'token', 'credential', 'credentials', 'api_key', 'private_key', 'security_code', 'otp', 'verification_code'}
_EMAIL_OR_URL_RE = re.compile('(?i)([A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}|https?://|www\\.|slack\\.com/|discord\\.gg/|\\+?\\d[\\d\\s().-]{8,})')
_SECRETISH_RE = re.compile('(?i)\\b(password|secret|token|credential|api[_-]?key|private[_-]?key|otp|verification\\s*code)\\b')

def _split_identifier(name: str) -> list[str]:
    return [part for part in re.split('[^a-zA-Z0-9]+', name.lower()) if part]

def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = value.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out

def _iter_strings(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_strings(nested)
    elif isinstance(value, str):
        text = value.strip()
        if text:
            yield text

def _infer_argument_roles(tool_args: dict | None) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    roles: list[str] = []
    destinations: list[str] = []
    selectors: list[str] = []
    temporal: list[str] = []
    payloads: list[str] = []
    sensitive: list[str] = []
    if not isinstance(tool_args, dict):
        return (roles, destinations, selectors, temporal, payloads, sensitive)
    for (key, value) in tool_args.items():
        key_l = key.lower()
        values = list(_iter_strings(value))
        if key_l in _DESTINATION_KEYS:
            roles.append('destination')
            destinations.extend(values)
        elif key_l in _INTERNAL_TARGET_KEYS:
            roles.append('selector')
            selectors.extend(values)
        elif key_l in _SELECTOR_KEYS or key_l.endswith('_name') or key_l.endswith('_names') or key_l.endswith('_id') or key_l.endswith('_ids'):
            roles.append('selector')
            selectors.extend(values)
        if key_l in _TEMPORAL_KEYS or key_l.endswith('_date') or key_l.endswith('_day') or key_l.endswith('_time'):
            roles.append('temporal')
            temporal.extend(values)
        if key_l in _PAYLOAD_KEYS:
            roles.append('payload')
            payloads.extend(values)
        if key_l in _SENSITIVE_KEYS:
            roles.append('sensitive')
            sensitive.extend(values)
    for text in list(destinations):
        if _EMAIL_OR_URL_RE.search(text):
            roles.append('external_target')
            if text not in destinations:
                destinations.append(text)
        if _SECRETISH_RE.search(text):
            roles.append('sensitive')
            sensitive.append(text)
    for text in list(payloads):
        if _SECRETISH_RE.search(text):
            roles.append('sensitive')
            sensitive.append(text)
    return (_dedupe_keep_order(roles), _dedupe_keep_order(destinations), _dedupe_keep_order(selectors), _dedupe_keep_order(temporal), _dedupe_keep_order(payloads), _dedupe_keep_order(sensitive))

def _infer_effect_type(tokens: set[str], tool_args: dict | None) -> tuple[str, list[str], list[str], list[str], list[str], list[str], list[str]]:
    (roles, destinations, selectors, temporal, payloads, sensitive) = _infer_argument_roles(tool_args)
    if tokens & _FINANCIAL_VERBS:
        return (EffectType.FINANCIAL, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _EXEC_VERBS:
        return (EffectType.CODE_EXECUTION, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _DELETE_VERBS:
        return (EffectType.DELETION, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _SHARE_VERBS:
        return (EffectType.SHARING, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _EXTERNAL_VERBS:
        return (EffectType.EXTERNAL_COMMUNICATION, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _COMMIT_VERBS:
        return (EffectType.COMMITMENT, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _SECURITY_ACCOUNT_VERBS:
        return (EffectType.SECURITY_ACCOUNT, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _WRITE_VERBS:
        commitment_shape = bool(tokens & _COMMITMENT_WRITE_VERBS) and bool(tokens & _COMMITMENT_OBJECT_NOUNS) and (bool(temporal) or bool(selectors) or bool(payloads) or bool(destinations))
        if commitment_shape:
            return (EffectType.COMMITMENT, roles, destinations, selectors, temporal, payloads, sensitive)
        if destinations:
            return (EffectType.EXTERNAL_COMMUNICATION, roles, destinations, selectors, temporal, payloads, sensitive)
        if selectors and any((token in tokens for token in {'schedule', 'book', 'invite', 'share', 'assign', 'connect'})):
            return (EffectType.COMMITMENT, roles, destinations, selectors, temporal, payloads, sensitive)
        if temporal or payloads:
            return (EffectType.STATE_MUTATION, roles, destinations, selectors, temporal, payloads, sensitive)
        return (EffectType.STATE_MUTATION, roles, destinations, selectors, temporal, payloads, sensitive)
    if tokens & _READ_VERBS:
        return (EffectType.NO_SIDE_EFFECT, roles, destinations, selectors, temporal, payloads, sensitive)
    if destinations:
        return (EffectType.EXTERNAL_COMMUNICATION, roles, destinations, selectors, temporal, payloads, sensitive)
    if selectors and temporal:
        return (EffectType.COMMITMENT, roles, destinations, selectors, temporal, payloads, sensitive)
    if payloads or temporal:
        return (EffectType.STATE_MUTATION, roles, destinations, selectors, temporal, payloads, sensitive)
    if selectors:
        return (EffectType.COMMITMENT, roles, destinations, selectors, temporal, payloads, sensitive)
    return (EffectType.UNKNOWN_HIGH_RISK, roles, destinations, selectors, temporal, payloads, sensitive)

def _effect_to_privilege(effect: str) -> str:
    if effect == EffectType.NO_SIDE_EFFECT:
        return PrivilegeLevel.READ
    if effect == EffectType.STATE_MUTATION:
        return PrivilegeLevel.WRITE
    return PrivilegeLevel.EXECUTE

class SourceAwareActionRouter:

    def __init__(self, custom_map: dict | None=None):
        self.privilege_map = dict(custom_map or {})
        self._stats = {'read_bypass': 0, 'guard_check': 0, 'guard_plus_isolator': 0}

    def profile_tool(self, tool_name: str, tool_args: dict | None=None) -> GateDecision:
        tokens = set(_split_identifier(tool_name))
        if tool_name in self.privilege_map:
            privilege = self.privilege_map[tool_name]
            effect = EffectType.NO_SIDE_EFFECT if privilege == PrivilegeLevel.READ else EffectType.UNKNOWN_HIGH_RISK
            (roles, destinations, selectors, temporal, payloads, sensitive) = _infer_argument_roles(tool_args)
        else:
            (effect, roles, destinations, selectors, temporal, payloads, sensitive) = _infer_effect_type(tokens, tool_args)
            privilege = _effect_to_privilege(effect)
        has_external_destination = bool(destinations and any((_EMAIL_OR_URL_RE.search(value) for value in destinations)))
        is_external_sink = effect in {EffectType.SHARING, EffectType.FINANCIAL} or has_external_destination
        is_high_consequence = privilege == PrivilegeLevel.EXECUTE or effect in {EffectType.DELETION, EffectType.SECURITY_ACCOUNT, EffectType.CODE_EXECUTION, EffectType.FINANCIAL, EffectType.COMMITMENT}
        if privilege == PrivilegeLevel.READ:
            action = 'auto_allow'
            reason = f'read-like action routed via {effect}'
        elif privilege == PrivilegeLevel.WRITE:
            action = 'guard_check'
            reason = f'state-changing action routed via {effect}'
        else:
            action = 'guard_plus_isolator'
            reason = f'high-consequence action routed via {effect}'
        return GateDecision(tool_name=tool_name, privilege=privilege, action=action, reason=reason, effect_type=effect, arg_roles=roles, destination_values=destinations, selector_values=selectors, temporal_values=temporal, target_values=_dedupe_keep_order(destinations + selectors), payload_values=payloads, sensitive_values=sensitive, is_external_sink=is_external_sink, is_high_consequence=is_high_consequence)

    def check(self, tool_name: str, tool_args: dict | None=None) -> GateDecision:
        decision = self.profile_tool(tool_name, tool_args)
        if decision.action == 'auto_allow':
            self._stats['read_bypass'] += 1
        elif decision.action == 'guard_check':
            self._stats['guard_check'] += 1
        else:
            self._stats['guard_plus_isolator'] += 1
        return decision

    def get_stats(self) -> dict:
        total = sum(self._stats.values())
        return {**self._stats, 'total': total, 'read_bypass_rate': round(self._stats['read_bypass'] / total, 3) if total else 0}

    def reset_stats(self):
        self._stats = {'read_bypass': 0, 'guard_check': 0, 'guard_plus_isolator': 0}
