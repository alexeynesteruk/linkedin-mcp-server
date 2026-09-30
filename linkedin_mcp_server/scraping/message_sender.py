"""Browser-UI message composition and send workflow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Literal
from urllib.parse import ParseResult, parse_qs, urljoin, urlparse

import anyio
import anyio.lowlevel
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import (
    InvalidReferenceError,
    LinkedInScraperException,
)
import linkedin_mcp_server.scraping.contracts as contracts
from linkedin_mcp_server.scraping.identifiers import (
    normalize_person_identifier,
    normalize_profile_urn,
    normalize_reply_thread_id,
    person_profile_url,
    reply_thread_path,
    reply_thread_url,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

_MESSAGING_COMPOSE_SELECTOR = '[role="textbox"][contenteditable="true"]'

_PROFILE_MESSAGE_TARGET_JS = r"""() => {
    const visible = element => {
        const visibility = element && getComputedStyle(element).visibility;
        return !!(
            element &&
            visibility !== 'hidden' &&
            visibility !== 'collapse' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    const active = anchor =>
        visible(anchor) &&
        !anchor.hasAttribute('disabled') &&
        (anchor.getAttribute('aria-disabled') || '').toLowerCase() !== 'true';
    const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
    const validComposeHref = value => {
        if (typeof value !== 'string' || /[\\\x00-\x1f\x7f]/.test(value)) {
            return false;
        }
        try {
            const url = new URL(value, window.location.href);
            const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
            if (
                url.protocol !== 'https:' ||
                !/(^|\.)linkedin\.com$/.test(hostname) ||
                url.username ||
                url.password ||
                (url.port && url.port !== '443') ||
                url.hash ||
                url.pathname !== '/messaging/compose/'
            ) {
                return false;
            }
            const values = [
                ...url.searchParams.getAll('recipient'),
                ...url.searchParams.getAll('profileUrn'),
            ];
            const normalized = values.map(item => {
                const text = item.trim();
                const prefix = 'urn:li:fsd_profile:';
                const identifier = text.startsWith(prefix)
                    ? text.slice(prefix.length)
                    : text;
                return /^[A-Za-z0-9_-]+$/.test(identifier) ? identifier : null;
            });
            return normalized.length > 0 &&
                normalized.every(item => item !== null && item === normalized[0]);
        } catch {
            return false;
        }
    };
    const main = document.querySelector('main');
    if (!main) return {status: 'unresolved'};

    // The top card is the first section that wraps no other section. LinkedIn
    // nests it inside a wrapper section and has moved the name between h1 and
    // h2, so neither its depth nor its heading level is pinned. It is chosen
    // before its heading is checked, so a card whose name has not rendered yet
    // stays unresolved instead of yielding to the next section. Sidebar
    // sections are skipped: they carry other people's Message links.
    const section = Array.from(main.querySelectorAll('section')).find(
        element =>
            visible(element) &&
            !element.closest('aside') &&
            !element.querySelector('section')
    );
    if (!section) return {status: 'unresolved'};
    const headings = Array.from(section.querySelectorAll('h1, h2, h3')).filter(visible);
    const visibleComposeAnchors = Array.from(
        section.querySelectorAll('a[href*="/messaging/compose/"]')
    ).filter(anchor => visible(anchor) && anchor.closest('section') === section);
    const composeAnchors = visibleComposeAnchors.filter(active);
    if (
        headings.length !== 1 ||
        composeAnchors.length > 1 ||
        (composeAnchors.length === 1 && visibleComposeAnchors.length !== 1)
    ) {
        return {status: 'unresolved'};
    }
    if (composeAnchors.length === 0) {
        return visibleComposeAnchors.length === 0
            ? {status: 'unavailable', pageUrl: window.location.href}
            : {status: 'unresolved'};
    }

    const anchor = composeAnchors[0];
    const composeHref = anchor.getAttribute('href') || anchor.href || '';
    if (!validComposeHref(composeHref)) return {status: 'unresolved'};
    return {
        status: 'resolved',
        pageUrl: window.location.href,
        displayName: normalize(
            headings[0].innerText || headings[0].textContent || ''
        ),
        composeHrefs: [composeHref],
    };
}"""

_PROFILE_MESSAGE_TARGET_READY_JS = (
    f"() => ({_PROFILE_MESSAGE_TARGET_JS})().status === 'resolved'"
)
_PROFILE_MESSAGE_TARGET_TIMEOUT_MS = 1_000
_MESSAGE_SUBMIT_READY_TIMEOUT_MS = 1_000
_MESSAGE_CLEANUP_TIMEOUT_SECONDS = 1.0

# A reply waits on the thread page itself, not on a compose surface. A long
# thread hydrates its history before its composer settles, and the user's
# fork recorded intermittent "Thread page did not load" on heavy React
# threads with the 5000ms default page timeout (fork commit 491b38d meant to
# raise it and committed no code). These are ceilings: a thread that settles
# sooner returns sooner, and every one of them expires before anything is
# typed or with the outcome reported as unconfirmed, never as a send.
_THREAD_READY_TIMEOUT_MS = 30_000
_THREAD_SUBMIT_READY_TIMEOUT_MS = 3_000
_THREAD_CONFIRMATION_TIMEOUT_MS = 15_000

# A thread target names a conversation, not a person, so no recipient identity
# exists to hold a local one against. The one reference the page offers is the
# pane's participant header: the profile links and recipient URNs the thread
# pane shows outside every message item and outside the composer's own scopes.
# A local identity the header does not also show is a contradiction, and so is
# one the check cannot read. Identities compare as bare identifiers, because a
# header link to /in/<URN>/ and a data-recipient-urn name the same member.
# Inlined after its host program defines visible, normalizeUrn and profilePath.
_MESSAGE_THREAD_IDENTITY_JS = r"""
    const threadMode = target => typeof target?.threadPath === 'string';
    const pathIdentifier = path =>
        typeof path === 'string' && path.startsWith('/in/') && path.endsWith('/')
            ? path.slice('/in/'.length, -1) || null
            : null;
    const paneHeaderIdentifiers = (pane, scopes, editor) => {
        const outside = element =>
            element !== editor &&
            !editor.contains(element) &&
            !element.closest('[data-view-name="message-list-item"]') &&
            !scopes.some(scope => scope.contains(element));
        const identifiers = new Set();
        for (const anchor of pane.querySelectorAll('a[href*="/in/"]')) {
            if (!visible(anchor) || !outside(anchor)) continue;
            const identifier = pathIdentifier(
                profilePath(anchor.getAttribute('href') || anchor.href || '')
            );
            if (identifier) identifiers.add(identifier);
        }
        for (const element of pane.querySelectorAll(
            '[data-profile-urn], [data-recipient-urn]'
        )) {
            if (!visible(element) || !outside(element)) continue;
            for (const name of ['data-profile-urn', 'data-recipient-urn']) {
                if (!element.hasAttribute(name)) continue;
                const identifier = normalizeUrn(element.getAttribute(name));
                if (identifier) identifiers.add(identifier);
            }
        }
        return identifiers;
    };
    const threadIdentitiesCorroborated = (paths, urns, header) =>
        paths.every(path => {
            const identifier = pathIdentifier(path);
            return identifier !== null && header.has(identifier);
        }) &&
        urns.every(urn => urn !== null && header.has(urn));
"""

# Narrow exception to the generic-selector rule for #1107: enterToSend uses
# the send-toggle class only when the verified composer has no Send button.
# If the class changes, confirmed sends remain unavailable.
_MESSAGE_COMPOSER_INSPECT_JS = (
    r"""
    const visible = element => {
        const visibility = element && getComputedStyle(element).visibility;
        return !!(
            element &&
            visibility !== 'hidden' &&
            visibility !== 'collapse' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    const normalizeUrn = value => {
        const text = (value || '').trim();
        const prefix = 'urn:li:fsd_profile:';
        const identifier = text.startsWith(prefix) ? text.slice(prefix.length) : text;
        return /^[A-Za-z0-9_-]+$/.test(identifier) ? identifier : null;
    };
    const profilePath = value => {
        if (typeof value !== 'string' || /[\\\x00-\x1f\x7f]/.test(value)) {
            return null;
        }
        try {
            const url = new URL(value, window.location.href);
            const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
            if (
                url.protocol !== 'https:' ||
                !/(^|\.)linkedin\.com$/.test(hostname) ||
                url.username ||
                url.password ||
                (url.port && url.port !== '443') ||
                url.hash
            ) {
                return null;
            }
            const match = /^\/in\/([^/?#]+)(?:\/.*)?$/.exec(url.pathname);
            return match ? `/in/${match[1]}/` : null;
        } catch {
            return null;
        }
    };
"""
    + _MESSAGE_THREAD_IDENTITY_JS
    + r"""
    const messageRoute = target => {
        try {
            const url = new URL(window.location.href);
            const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
            if (
                url.protocol !== 'https:' ||
                !/(^|\.)linkedin\.com$/.test(hostname) ||
                url.username ||
                url.password ||
                (url.port && url.port !== '443') ||
                url.hash ||
                !(
                    url.pathname === '/messaging/compose/' ||
                    /^\/messaging\/thread\/[A-Za-z0-9_=-]+\/$/.test(url.pathname)
                )
            ) {
                return null;
            }
            const values = [
                ...url.searchParams.getAll('recipient'),
                ...url.searchParams.getAll('profileUrn'),
            ];
            if (threadMode(target)) {
                // A thread route may name no recipient: there is none here to
                // hold one against, and the path alone is the target.
                return url.pathname === target.threadPath && values.length === 0
                    ? url.href
                    : null;
            }
            return values.every(value => normalizeUrn(value) === target.profileUrn)
                ? url.href
                : null;
        } catch {
            return null;
        }
    };
    const inspect = target => {
        const editors = Array.from(
            document.querySelectorAll('[role="textbox"][contenteditable="true"]')
        ).filter(visible);
        if (editors.length !== 1) return {status: 'ambiguous_editor'};
        const editor = editors[0];
        const semanticAncestors = element => {
            const scopes = [];
            let ancestor = element.parentElement;
            while (ancestor) {
                if (ancestor.matches('form, dialog, [role="dialog"]')) {
                    scopes.push(ancestor);
                }
                ancestor = ancestor.parentElement;
            }
            return scopes;
        };
        const localScopes = semanticAncestors(editor);
        if (localScopes.length === 0) return {status: 'missing_owner'};

        const owner = localScopes.find(scope =>
            scope.matches('dialog, [role="dialog"]')
        ) || localScopes[0];
        // A thread is answered from its own pane on the full messaging page:
        // the nearest ancestor of the composer that holds a visible message.
        // Until one renders the conversation has not hydrated and nothing is
        // proven yet. An overlay chat belongs to whichever conversation it was
        // opened for, and it never qualifies: a dialog above the editor is the
        // owner, and threadScope never climbs out of a dialog.
        let pane = null;
        if (threadMode(target)) {
            pane = threadScope(owner);
            if (
                pane === owner ||
                !Array.from(
                    pane.querySelectorAll('[data-view-name="message-list-item"]')
                ).some(visible)
            ) {
                return {status: 'thread_pending'};
            }
        }
        const outsideDraftAndHistory = element =>
            element !== editor &&
            !editor.contains(element) &&
            !element.closest('[data-view-name="message-list-item"]');
        const identityElements = selector => Array.from(new Set(
            localScopes.flatMap(scope => [
                ...(scope.matches(selector) ? [scope] : []),
                ...scope.querySelectorAll(selector),
            ])
        ));
        const paths = identityElements('a[href*="/in/"]')
            .filter(element => visible(element) && outsideDraftAndHistory(element))
            .map(anchor => profilePath(anchor.getAttribute('href') || anchor.href || ''));
        const urns = identityElements(
            '[data-profile-urn], [data-recipient-urn]'
        ).filter(
            element => visible(element) && outsideDraftAndHistory(element)
        ).flatMap(element =>
            ['data-profile-urn', 'data-recipient-urn']
                .filter(name => element.hasAttribute(name))
                .map(name => normalizeUrn(element.getAttribute(name)))
        );
        if (
            threadMode(target)
                ? !threadIdentitiesCorroborated(
                    paths, urns, paneHeaderIdentifiers(pane, localScopes, editor)
                )
                : paths.some(path => path !== target.profilePath) ||
                    urns.some(urn => urn !== target.profileUrn)
        ) {
            return {status: 'recipient_mismatch'};
        }

        const submitButtons = scope => Array.from(
            scope.querySelectorAll(
                'button[type="submit"], button[data-control-name="send"]'
            )
        ).filter(button =>
            visible(button) &&
            !button.closest('[data-view-name="message-list-item"]')
        );
        const localScope = localScopes.find(scope => submitButtons(scope).length > 0)
            || localScopes[0];
        const buttons = submitButtons(localScope);
        // With LinkedIn's "Press Enter to Send" preference the composer
        // renders no Send button, only the send-options toggle.
        const enterToSend = buttons.length === 0 && localScopes.some(scope =>
            Array.from(scope.querySelectorAll('.msg-form__send-toggle')).some(visible)
        );
        return {
            status: 'valid',
            editor,
            ancestorChain: localScopes,
            localScope,
            owner,
            buttons,
            enterToSend,
            active: document.activeElement === editor,
            empty: !(editor.innerText || '').replace(/\s+/g, ' ').trim(),
            messageRoute: messageRoute(target),
            pane,
        };
    };
    // The element whose subtree holds this conversation's messages. An
    // overlay dialog holds both the messages and the composer. On the full
    // messaging page the owner is the composer <form> and the messages are
    // in its sibling, so climb to the nearest ancestor that holds a message
    // item, one visible editor, and stays below <main>. Otherwise keep the
    // owner, which leaves the send unconfirmed rather than widening the scope.
    const threadScope = owner => {
        if (!owner || owner.matches('dialog, [role="dialog"]')) return owner;
        const lists = '[data-view-name="message-list-item"]';
        let ancestor = owner.parentElement;
        while (ancestor && !ancestor.matches('main, body')) {
            const editors = Array.from(ancestor.querySelectorAll(
                '[role="textbox"][contenteditable="true"]'
            )).filter(visible);
            if (editors.length !== 1) return owner;
            if (ancestor.querySelector(lists)) return ancestor;
            ancestor = ancestor.parentElement;
        }
        return owner;
    };
"""
)

_MESSAGE_COMPOSER_OWNER_JS = (
    "(arg) => {"
    + _MESSAGE_COMPOSER_INSPECT_JS
    + """
        const target = arg.target;
        const state = inspect(target);
        if (
            state.status !== 'valid' ||
            state.messageRoute !== arg.expectedRoute ||
            !state.owner.isConnected ||
            !state.editor.isConnected ||
            !state.owner.contains(state.editor) ||
            state.buttons.length !== 1
        ) {
            return null;
        }
        const button = state.buttons[0];
        if (
            !button.isConnected ||
            !state.localScope.contains(button) ||
            (button.form !== null && !state.ancestorChain.includes(button.form))
        ) {
            return null;
        }
        state.owner.__linkedinMcpComposer = {
            editor: state.editor,
            ancestorChain: state.ancestorChain,
            button,
            localScope: state.localScope,
            profilePath: target.profilePath,
            profileUrn: target.profileUrn,
            threadPath: threadMode(target) ? target.threadPath : null,
            pane: state.pane,
            route: arg.expectedRoute,
            ownedMessage: null,
        };
        return state.owner;
    }"""
)

_MESSAGE_CONFIRMATION_PREPARE_JS = (
    "(arg) => {"
    + _MESSAGE_COMPOSER_INSPECT_JS
    + r"""
        const composer = inspect(arg);
        const pinned = arg.owner?.__linkedinMcpComposer;
        if (
            composer.status !== 'valid' ||
            composer.messageRoute !== pinned?.route ||
            !pinned ||
            pinned.threadPath !== (threadMode(arg) ? arg.threadPath : null) ||
            composer.pane !== pinned.pane ||
            composer.owner !== arg.owner ||
            composer.editor !== pinned.editor ||
            composer.ancestorChain.length !== pinned.ancestorChain.length ||
            composer.ancestorChain.some(
                (scope, index) => scope !== pinned.ancestorChain[index]
            ) ||
            composer.localScope !== pinned.localScope ||
            composer.buttons.length !== 1 ||
            composer.buttons[0] !== pinned.button ||
            pinned.button.disabled ||
            (pinned.button.getAttribute('aria-disabled') || '').toLowerCase()
                === 'true' ||
            !arg.owner.isConnected ||
            !pinned.editor.isConnected ||
            !arg.owner.contains(pinned.editor) ||
            document.activeElement !== pinned.editor ||
            pinned.ownedMessage !== arg.expected ||
            (pinned.editor.innerText || pinned.editor.textContent || '') !== arg.expected
        ) {
            return null;
        }

        const counter = (arg.owner.__linkedinMcpConfirmationCounter || 0) + 1;
        arg.owner.__linkedinMcpConfirmationCounter = counter;
        const token = String(counter);
        const marker = document.createElement('span');
        marker.hidden = true;
        marker.setAttribute('data-linkedin-mcp-confirmation', token);
        marker.setAttribute('data-linkedin-mcp-invalid', 'false');
        arg.owner.appendChild(marker);
        pinned.editor.setAttribute('data-linkedin-mcp-editor', token);
        const state = {
            owner: arg.owner,
            scope: threadScope(arg.owner),
            editor: pinned.editor,
            expected: arg.expected,
            baseline: new Set(),
            candidates: new Map(),
            invalid: false,
        };
        const exactUnit = (node, requireVisible) => {
            if (requireVisible && !visible(node)) return false;
            const elements = [node, ...node.querySelectorAll('*')].filter(
                element => !requireVisible || visible(element)
            );
            const matches = elements.filter(
                element => (element.innerText || '') === state.expected
            );
            const smallest = matches.filter(
                element => !matches.some(
                    other => other !== element && element.contains(other)
                )
            );
            return smallest.length === 1;
        };
        const remember = node => {
            if (!(node instanceof Element)) return;
            const items = [
                ...(node.matches('[data-view-name="message-list-item"]')
                    ? [node]
                    : []),
                ...node.querySelectorAll('[data-view-name="message-list-item"]'),
            ];
            for (const item of items) {
                if (state.baseline.has(item)) continue;
                if (!state.candidates.has(item)) {
                    item.setAttribute('data-linkedin-mcp-candidate', token);
                    state.candidates.set(item, {
                        transitioned: false,
                        matched: false,
                    });
                }
            }
        };
        const refresh = () => {
            for (const [node, candidate] of state.candidates) {
                if (
                    node.isConnected &&
                    state.scope.contains(node) &&
                    exactUnit(node, true)
                ) {
                    candidate.matched = true;
                    node.setAttribute('data-linkedin-mcp-matched', token);
                }
                if (candidate.matched && !node.isConnected) {
                    state.invalid = true;
                    marker.setAttribute('data-linkedin-mcp-invalid', 'true');
                }
            }
            if (
                Array.from(state.candidates.values()).filter(
                    candidate => candidate.matched
                ).length > 1
            ) {
                state.invalid = true;
                marker.setAttribute('data-linkedin-mcp-invalid', 'true');
            }
        };
        state.observer = new MutationObserver(records => {
            for (const record of records) {
                if (record.type !== 'childList') continue;
                for (const node of record.addedNodes) remember(node);
                for (const removed of record.removedNodes) {
                    if (!(removed instanceof Element)) continue;
                    if (removed === state.editor || removed.contains(state.editor)) {
                        state.invalid = true;
                        marker.setAttribute('data-linkedin-mcp-invalid', 'true');
                    }
                    for (const [candidate, entry] of state.candidates) {
                        if (
                            (removed === candidate || removed.contains(candidate)) &&
                            exactUnit(candidate, false)
                        ) {
                            state.invalid = true;
                            marker.setAttribute(
                                'data-linkedin-mcp-invalid', 'true'
                            );
                            // LinkedIn's own rendering of this submission
                            // (measured, #1108): a node inserted after it, seen
                            // showing exactly the text under a client-side ID,
                            // then taken away as the server copy replaced it.
                            // An attribute, since the readiness check runs in
                            // another world. Only a thread reply requires it.
                            if (
                                entry.matched &&
                                !(candidate.getAttribute('data-event-urn') || '')
                                    .trim()
                                    .startsWith('urn:li:msg_message:')
                            ) {
                                marker.setAttribute(
                                    'data-linkedin-mcp-placeholder', 'true'
                                );
                            }
                        }
                    }
                }
            }
            for (const record of records) {
                if (
                    record.type !== 'attributes' ||
                    !state.candidates.has(record.target)
                ) {
                    continue;
                }
                const before = (record.oldValue || '').trim();
                const after = (
                    record.target.getAttribute('data-event-urn') || ''
                ).trim();
                if (before && after && before !== after) {
                    state.candidates.get(record.target).transitioned = true;
                    record.target.setAttribute(
                        'data-linkedin-mcp-transitioned', token
                    );
                }
            }
            refresh();
        });
        state.baseline = new Set(
            document.querySelectorAll('[data-view-name="message-list-item"]')
        );
        // Kept as an attribute: the readiness check runs in another world,
        // where properties set on elements here are not visible.
        marker.setAttribute('data-linkedin-mcp-route', window.location.pathname);
        marker.setAttribute('data-linkedin-mcp-baseline', JSON.stringify(
            Array.from(state.baseline)
                .map(node => (node.getAttribute('data-event-urn') || '').trim())
                .filter(Boolean)
        ));
        state.observer.observe(state.scope, {
            attributes: true,
            attributeFilter: ['data-event-urn'],
            attributeOldValue: true,
            childList: true,
            subtree: true,
        });
        if (!arg.owner.__linkedinMcpConfirmations) {
            arg.owner.__linkedinMcpConfirmations = new Map();
        }
        arg.owner.__linkedinMcpConfirmations.set(token, state);
        return token;
    }"""
)

_MESSAGE_CONFIRMATION_READY_JS = (
    "(arg) => {"
    + _MESSAGE_COMPOSER_INSPECT_JS
    + r"""
        const exactVisibleUnit = node => {
            if (!visible(node)) return false;
            const elements = [node, ...node.querySelectorAll('*')].filter(visible);
            const matches = elements.filter(
                element => (element.innerText || '') === arg.expected
            );
            return matches.filter(
                element => !matches.some(
                    other => other !== element && element.contains(other)
                )
            ).length === 1;
        };
        const itemSelector = '[data-view-name="message-list-item"]';
        const linksRecipient = anchor => {
            let path;
            try {
                path = new URL(
                    anchor.getAttribute('href') || '', window.location.href
                ).pathname;
            } catch {
                return false;
            }
            const identifier = /^\/in\/([^/]+)/.exec(path)?.[1];
            return !!identifier && (
                identifier === arg.profileUrn || `/in/${identifier}/` === arg.profilePath
            );
        };
        // LinkedIn heads a message with links to its sender's profile
        // (measured). A follow-up from the same sender is assumed to be
        // unheaded, so the sender of a node is the nearest item at or before
        // it that links a profile. A
        // message from the recipient can carry the same text without this
        // submission ever reaching LinkedIn, so refuse a node the recipient
        // sent, or one whose sender cannot be found. Only the link path
        // counts, never its text.
        const sentByRecipient = (scope, node) => {
            const all = Array.from(scope.querySelectorAll(itemSelector));
            const sender = all.slice(0, all.indexOf(node) + 1).reverse().find(
                item => item.querySelector('a[href*="/in/"]')
            );
            return !sender || Array.from(
                sender.querySelectorAll('a[href*="/in/"]')
            ).some(linksRecipient);
        };
        // A thread target knows no recipient, so the other side is whoever
        // the pane's participant header names. The sender is found as above;
        // a node whose sender cannot be found, whose header link cannot be
        // read, or who is named in the pane header, is not this submission.
        // A header that names nobody refuses nothing here, which is why a
        // thread acknowledgement also needs LinkedIn's local placeholder.
        const sentByParticipant = (scope, node, composer) => {
            const all = Array.from(scope.querySelectorAll(itemSelector));
            const sender = all.slice(0, all.indexOf(node) + 1).reverse().find(
                item => item.querySelector('a[href*="/in/"]')
            );
            if (!sender) return true;
            const header = paneHeaderIdentifiers(
                scope, composer.ancestorChain, composer.editor
            );
            return Array.from(sender.querySelectorAll('a[href*="/in/"]')).some(
                anchor => {
                    const identifier = pathIdentifier(
                        profilePath(anchor.getAttribute('href') || '')
                    );
                    return identifier === null || header.has(identifier);
                }
            );
        };
        // LinkedIn acknowledges a send by rendering a node whose event ID is
        // a server message URN. In an open thread it inserts that node and
        // removes its client-side placeholder; the first message of a new
        // thread moves the route to /messaging/thread/<id>/ and remounts the
        // whole conversation pane, composer included. Neither keeps the
        // observed node or the pinned composer, so accept exactly one visible
        // exact-text node carrying a server URN that was absent before submit,
        // inside the conversation pane of the one composer the page now shows.
        // That node must be the newest message in the pane, so older history
        // loaded later cannot stand in for it. A send that started on a
        // thread route must stay on that thread. One that started on the
        // compose route and now sits on a thread route is the first message
        // of a new thread, whose pane holds that one message; a pane with
        // other messages there is some other conversation, and so is one
        // whose header does not link the recipient.
        const serverAcknowledged = () => {
            const marker = Array.from(
                arg.owner?.querySelectorAll('[data-linkedin-mcp-confirmation]') || []
            ).find(
                node => node.getAttribute('data-linkedin-mcp-confirmation') === arg.token
            );
            if (!marker?.hasAttribute('data-linkedin-mcp-baseline')) return false;
            let baselineUrns;
            try {
                baselineUrns = new Set(
                    JSON.parse(marker?.getAttribute('data-linkedin-mcp-baseline'))
                );
            } catch {
                return false;
            }
            const composer = inspect(arg);
            if (composer.status !== 'valid' || composer.messageRoute === null) {
                return false;
            }
            const scope = threadScope(composer.owner);
            const items = Array.from(
                scope.querySelectorAll(itemSelector)
            ).filter(visible);
            const startPath = marker.getAttribute('data-linkedin-mcp-route') || '';
            const path = window.location.pathname;
            if (threadMode(arg)) {
                // A reply never moves: it started on the thread the caller
                // named and is acknowledged there, after LinkedIn rendered
                // this submission locally first.
                if (
                    path !== arg.threadPath ||
                    startPath !== arg.threadPath ||
                    marker.getAttribute('data-linkedin-mcp-placeholder') !== 'true'
                ) {
                    return false;
                }
            } else if (path.startsWith('/messaging/thread/')) {
                if (startPath.startsWith('/messaging/thread/')) {
                    if (path !== startPath) return false;
                } else if (
                    items.length !== 1 ||
                    !Array.from(scope.querySelectorAll('a[href*="/in/"]')).some(
                        anchor => !anchor.closest(itemSelector) &&
                            linksRecipient(anchor)
                    )
                ) {
                    return false;
                }
            }
            const acknowledged = items.filter(node => {
                const urn = (node.getAttribute('data-event-urn') || '').trim();
                return urn.startsWith('urn:li:msg_message:') &&
                    !baselineUrns.has(urn) &&
                    exactVisibleUnit(node);
            });
            return acknowledged.length === 1 &&
                acknowledged[0] === items[items.length - 1] &&
                !(threadMode(arg)
                    ? sentByParticipant(scope, acknowledged[0], composer)
                    : sentByRecipient(scope, acknowledged[0]));
        };
        if (!arg.owner?.isConnected) return serverAcknowledged();
        const markers = Array.from(
            arg.owner.querySelectorAll('[data-linkedin-mcp-confirmation]')
        ).filter(
            marker => marker.getAttribute('data-linkedin-mcp-confirmation') === arg.token
        );
        if (
            markers.length !== 1 ||
            markers[0].getAttribute('data-linkedin-mcp-invalid') !== 'false'
        ) {
            return serverAcknowledged();
        }
        const composer = inspect(arg);
        if (
            composer.status !== 'valid' ||
            composer.messageRoute === null ||
            composer.owner !== arg.owner ||
            composer.buttons.length !== 1 ||
            composer.editor.getAttribute('data-linkedin-mcp-editor') !== arg.token
        ) {
            return serverAcknowledged();
        }
        const candidates = Array.from(
            threadScope(arg.owner).querySelectorAll('[data-linkedin-mcp-candidate]')
        ).filter(node =>
            node.getAttribute('data-linkedin-mcp-candidate') === arg.token &&
            node.getAttribute('data-linkedin-mcp-matched') === arg.token &&
            node.getAttribute('data-linkedin-mcp-transitioned') === arg.token &&
            (node.getAttribute('data-event-urn') || '').trim() &&
            exactVisibleUnit(node)
        );
        return candidates.length === 1 || serverAcknowledged();
    }"""
)

_MESSAGE_CONFIRMATION_DISPOSE_JS = r"""arg => {
    const confirmations = arg.owner?.__linkedinMcpConfirmations;
    const state = confirmations?.get(arg.token);
    if (state?.observer) state.observer.disconnect();
    confirmations?.delete(arg.token);
    for (const element of (state?.scope || arg.owner)?.querySelectorAll(
        '[data-linkedin-mcp-candidate], [data-linkedin-mcp-editor], '
        + '[data-linkedin-mcp-confirmation]'
    ) || []) {
        for (const attribute of [
            'data-linkedin-mcp-candidate',
            'data-linkedin-mcp-matched',
            'data-linkedin-mcp-transitioned',
            'data-linkedin-mcp-editor',
        ]) {
            if (element.getAttribute(attribute) === arg.token) {
                element.removeAttribute(attribute);
            }
        }
        if (element.getAttribute('data-linkedin-mcp-confirmation') === arg.token) {
            element.remove();
        }
    }
}"""

_MESSAGE_COMPOSER_DISPOSE_JS = r"""owner => {
    const confirmations = owner?.__linkedinMcpConfirmations;
    const scopes = new Set([owner]);
    for (const state of confirmations?.values() || []) {
        if (state?.observer) state.observer.disconnect();
        if (state?.scope) scopes.add(state.scope);
    }
    confirmations?.clear();
    if (owner) {
        delete owner.__linkedinMcpConfirmations;
        delete owner.__linkedinMcpComposer;
    }
    const marked = Array.from(scopes).flatMap(scope => Array.from(
        scope?.querySelectorAll(
            '[data-linkedin-mcp-candidate], [data-linkedin-mcp-editor], '
            + '[data-linkedin-mcp-confirmation]'
        ) || []
    ));
    for (const element of new Set(marked)) {
        element.removeAttribute('data-linkedin-mcp-candidate');
        element.removeAttribute('data-linkedin-mcp-matched');
        element.removeAttribute('data-linkedin-mcp-transitioned');
        element.removeAttribute('data-linkedin-mcp-editor');
        if (element.hasAttribute('data-linkedin-mcp-confirmation')) element.remove();
    }
}"""

_MESSAGE_COMPOSER_STATE_JS = (
    "(target) => {"
    + _MESSAGE_COMPOSER_INSPECT_JS
    + """
        const state = inspect(target);
        return {
            status: state.status,
            active: state.active === true,
            empty: state.empty === true,
            submitCount: state.buttons ? state.buttons.length : 0,
            enterToSend: state.enterToSend === true,
            submitUsable: state.buttons?.length === 1 &&
                !state.buttons[0].disabled &&
                (state.buttons[0].getAttribute('aria-disabled') || '').toLowerCase()
                    !== 'true',
        };
    }"""
)

_MESSAGE_COMPOSER_READY_JS = (
    "(target) => {"
    + _MESSAGE_COMPOSER_INSPECT_JS
    + """
        return inspect(target).status === 'valid';
    }"""
)

_MESSAGE_COMPOSER_FOCUS_JS = (
    "(target) => {"
    + _MESSAGE_COMPOSER_INSPECT_JS
    + """
        const state = inspect(target);
        if (state.status !== 'valid') return false;
        state.editor.focus();
        return state.editor.isConnected && document.activeElement === state.editor;
    }"""
)

_MESSAGE_COMPOSER_PINNED_JS = (
    r"""
    const visible = element => {
        const visibility = element && getComputedStyle(element).visibility;
        return !!(
            element &&
            visibility !== 'hidden' &&
            visibility !== 'collapse' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    const normalizeUrn = value => {
        const text = (value || '').trim();
        const prefix = 'urn:li:fsd_profile:';
        const identifier = text.startsWith(prefix) ? text.slice(prefix.length) : text;
        return /^[A-Za-z0-9_-]+$/.test(identifier) ? identifier : null;
    };
    const profilePath = value => {
        if (typeof value !== 'string' || /[\\\x00-\x1f\x7f]/.test(value)) {
            return null;
        }
        try {
            const url = new URL(value, window.location.href);
            const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
            if (
                url.protocol !== 'https:' ||
                !/(^|\.)linkedin\.com$/.test(hostname) ||
                url.username ||
                url.password ||
                (url.port && url.port !== '443') ||
                url.hash
            ) {
                return null;
            }
            const match = /^\/in\/([^/?#]+)(?:\/.*)?$/.exec(url.pathname);
            return match ? `/in/${match[1]}/` : null;
        } catch {
            return null;
        }
    };
"""
    + _MESSAGE_THREAD_IDENTITY_JS
    + r"""
    const messageRoute = target => {
        try {
            const url = new URL(window.location.href);
            const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
            if (
                url.protocol !== 'https:' ||
                !/(^|\.)linkedin\.com$/.test(hostname) ||
                url.username ||
                url.password ||
                (url.port && url.port !== '443') ||
                url.hash ||
                !(
                    url.pathname === '/messaging/compose/' ||
                    /^\/messaging\/thread\/[A-Za-z0-9_=-]+\/$/.test(url.pathname)
                )
            ) {
                return null;
            }
            const values = [
                ...url.searchParams.getAll('recipient'),
                ...url.searchParams.getAll('profileUrn'),
            ];
            if (threadMode(target)) {
                return url.pathname === target.threadPath && values.length === 0
                    ? url.href
                    : null;
            }
            return values.every(value => normalizeUrn(value) === target.profileUrn)
                ? url.href
                : null;
        } catch {
            return null;
        }
    };
    const semanticAncestors = element => {
        const scopes = [];
        let ancestor = element?.parentElement;
        while (ancestor) {
            if (ancestor.matches('form, dialog, [role="dialog"]')) {
                scopes.push(ancestor);
            }
            ancestor = ancestor.parentElement;
        }
        return scopes;
    };
    const identitiesMatch = (scopes, editor, target, pane) => {
        const outsideDraftAndHistory = element =>
            element !== editor &&
            !editor.contains(element) &&
            !element.closest('[data-view-name="message-list-item"]');
        const identityElements = selector => Array.from(new Set(
            scopes.flatMap(scope => [
                ...(scope.matches(selector) ? [scope] : []),
                ...scope.querySelectorAll(selector),
            ])
        ));
        const paths = identityElements('a[href*="/in/"]')
            .filter(element => visible(element) && outsideDraftAndHistory(element))
            .map(anchor => profilePath(anchor.getAttribute('href') || anchor.href || ''));
        const urns = identityElements(
            '[data-profile-urn], [data-recipient-urn]'
        ).filter(
            element => visible(element) && outsideDraftAndHistory(element)
        ).flatMap(element =>
            ['data-profile-urn', 'data-recipient-urn']
                .filter(name => element.hasAttribute(name))
                .map(name => normalizeUrn(element.getAttribute(name)))
        );
        if (threadMode(target)) {
            // The pane pinned with the composer: PREPARE refuses a submission
            // once the composer sits in any other.
            return !!pane && threadIdentitiesCorroborated(
                paths, urns, paneHeaderIdentifiers(pane, scopes, editor)
            );
        }
        return !(
            paths.some(path => path !== target.profilePath) ||
            urns.some(urn => urn !== target.profileUrn)
        );
    };
    const validatePinned = (target, requireEnabled = true) => {
        const pinned = owner?.__linkedinMcpComposer;
        if (
            !pinned ||
            pinned.profilePath !== target.profilePath ||
            pinned.profileUrn !== target.profileUrn ||
            pinned.threadPath !== (threadMode(target) ? target.threadPath : null) ||
            messageRoute(target) !== pinned.route
        ) {
            return null;
        }
        const {editor, ancestorChain, button, localScope, pane} = pinned;
        const currentChain = semanticAncestors(editor);
        if (
            !owner.isConnected ||
            !editor?.isConnected ||
            !button?.isConnected ||
            !localScope?.isConnected ||
            !Array.isArray(ancestorChain) ||
            currentChain.length !== ancestorChain.length ||
            currentChain.some((scope, index) => scope !== ancestorChain[index]) ||
            !currentChain.includes(owner) ||
            !currentChain.includes(localScope) ||
            !owner.contains(editor) ||
            !owner.contains(localScope) ||
            !localScope.contains(button) ||
            (button.form !== null && !currentChain.includes(button.form)) ||
            !visible(editor) ||
            !visible(button) ||
            !editor.matches('[role="textbox"][contenteditable="true"]') ||
            !identitiesMatch(currentChain, editor, target, pane)
        ) {
            return null;
        }
        const buttons = Array.from(localScope.querySelectorAll(
            'button[type="submit"], button[data-control-name="send"]'
        )).filter(candidate =>
            visible(candidate) &&
            !candidate.closest('[data-view-name="message-list-item"]')
        );
        if (
            buttons.length !== 1 ||
            buttons[0] !== button ||
            (requireEnabled && (
                button.disabled ||
                (button.getAttribute('aria-disabled') || '').toLowerCase() === 'true'
            ))
        ) {
            return null;
        }
        return pinned;
    };
"""
)

_MESSAGE_COMPOSER_WRITE_JS = (
    "(owner, arg) => {"
    + _MESSAGE_COMPOSER_PINNED_JS
    + r"""
        let pinned = validatePinned(arg, false);
        if (!pinned) return 'invalid';
        const {editor} = pinned;
        if ((editor.innerText || '').replace(/\s+/g, ' ').trim()) {
            return 'occupied';
        }
        editor.focus();
        pinned = validatePinned(arg, false);
        if (!pinned || document.activeElement !== editor) return 'invalid';
        if ((editor.innerText || '').replace(/\s+/g, ' ').trim()) {
            return 'occupied';
        }
        if (
            typeof document.queryCommandSupported !== 'function' ||
            !document.queryCommandSupported('insertText') ||
            typeof document.execCommand !== 'function'
        ) {
            return 'unsupported';
        }
        const inserted = document.execCommand('insertText', false, arg.message);
        if ((editor.innerText || editor.textContent || '') === arg.message) {
            pinned.ownedMessage = arg.message;
        }
        if (inserted !== true) return 'unsupported';
        pinned = validatePinned(arg, false);
        if (
            !pinned ||
            document.activeElement !== editor ||
            pinned.ownedMessage !== arg.message ||
            (editor.innerText || editor.textContent || '') !== arg.message
        ) {
            return 'invalid';
        }
        return 'written';
    }"""
)

_MESSAGE_COMPOSER_SUBMIT_READY_JS = (
    "(owner, arg) => {"
    + _MESSAGE_COMPOSER_PINNED_JS
    + r"""
        const pinned = validatePinned(arg, false);
        if (
            !pinned ||
            document.activeElement !== pinned.editor ||
            pinned.ownedMessage !== arg.message ||
            (pinned.editor.innerText || pinned.editor.textContent || '') !== arg.message
        ) {
            return 'invalid';
        }
        return pinned.button.disabled ||
            (pinned.button.getAttribute('aria-disabled') || '').toLowerCase() === 'true'
            ? 'disabled'
            : 'ready';
    }"""
)

_MESSAGE_COMPOSER_CLEANUP_JS = r"""(owner, arg) => {
    const pinned = owner?.__linkedinMcpComposer;
    if (!pinned || pinned.ownedMessage !== arg.message) return false;
    const {editor, ancestorChain} = pinned;
    const currentChain = [];
    let ancestor = editor?.parentElement;
    while (ancestor) {
        if (ancestor.matches('form, dialog, [role="dialog"]')) {
            currentChain.push(ancestor);
        }
        ancestor = ancestor.parentElement;
    }
    if (
        !owner.isConnected ||
        !editor?.isConnected ||
        !Array.isArray(ancestorChain) ||
        currentChain.length !== ancestorChain.length ||
        currentChain.some((scope, index) => scope !== ancestorChain[index]) ||
        !currentChain.includes(owner) ||
        !owner.contains(editor) ||
        (editor.innerText || editor.textContent || '') !== arg.message
    ) {
        return false;
    }
    pinned.ownedMessage = null;
    editor.replaceChildren();
    editor.dispatchEvent(new InputEvent('input', {
        bubbles: true,
        composed: true,
        data: null,
        inputType: 'deleteContentBackward',
    }));
    return true;
}"""

_MESSAGE_COMPOSER_SUBMIT_JS = (
    "(owner, arg) => {"
    + _MESSAGE_COMPOSER_PINNED_JS
    + r"""
        const pinned = validatePinned(arg);
        if (
            !pinned ||
            document.activeElement !== pinned.editor ||
            pinned.ownedMessage !== arg.message ||
            (pinned.editor.innerText || pinned.editor.textContent || '') !== arg.message
        ) {
            return 'invalid';
        }
        pinned.button.click();
        return 'clicked';
    }"""
)

_LINKEDIN_MESSAGE_HOST_RE = re.compile(r"^(?:[a-z0-9-]+\.)*linkedin\.com$")
_PROFILE_PATH_RE = re.compile(r"^/in/[^/?#]+/$")
# A thread id is base64url and keeps its padding literally. Measured live:
# /messaging/thread/2-ZDBkMjZiY2Ut...XzEwMA==/ is what LinkedIn redirects an
# existing conversation to, and rejecting it stopped every send to a member
# the account had already written to. Only '=' is added: '%' would readmit an
# encoded slash and let one path pose as another. The id identifies nobody on
# its own, and the recipient is proven by the composer rather than this path.
# A thread reply is the exception by design: there the caller named the
# conversation, so this exact path is the target (see _ThreadMessageTarget),
# and identifiers.normalize_reply_thread_id admits the same alphabet.
_MESSAGE_THREAD_PATH_RE = re.compile(r"^/messaging/thread/[A-Za-z0-9_=-]+/$")
_PROFILE_URN_PREFIX = "urn:li:fsd_profile:"


@dataclass(frozen=True)
class _ProfileMessageTarget:
    profile_path: str
    profile_urn: str
    compose_url: str
    display_name: str | None


@dataclass(frozen=True)
class _ProfileMessageTargetResolution:
    status: Literal["resolved", "unavailable", "failed"]
    target: _ProfileMessageTarget | None = None


@dataclass(frozen=True)
class _ThreadMessageTarget:
    """An existing conversation, named by the thread id the caller passed.

    It names no person. The route is the whole target: the page JS pins
    ``thread_path`` exactly, accepts only a composer docked in that thread's
    own pane, and holds any local identity against the pane's participant
    header, the one reference a thread page offers.
    """

    thread_id: str
    thread_path: str
    thread_url: str


_MessageTarget = _ProfileMessageTarget | _ThreadMessageTarget


def _thread_message_target(thread_id: str) -> _ThreadMessageTarget:
    """Validate a caller's thread id into the route a reply is pinned to."""
    thread_id = normalize_reply_thread_id(thread_id)
    return _ThreadMessageTarget(
        thread_id=thread_id,
        thread_path=reply_thread_path(thread_id),
        thread_url=reply_thread_url(thread_id),
    )


def _safe_linkedin_url(value: str, *, base: str | None = None) -> ParseResult | None:
    """Parse an HTTPS LinkedIn URL without credentials or an ambiguous origin."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return None
    candidate = urljoin(base, value.strip()) if base else value.strip()
    try:
        parsed = urlparse(candidate)
        port = parsed.port
    except ValueError:
        return None
    hostname = (parsed.hostname or "").lower().removesuffix(".")
    if (
        parsed.scheme != "https"
        or not _LINKEDIN_MESSAGE_HOST_RE.fullmatch(hostname)
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        return None
    return parsed


def _normalize_profile_urn(value: str | None) -> str | None:
    """Return the identifier carried by a profile URN or raw recipient value."""
    if not isinstance(value, str):
        return None
    try:
        candidate = normalize_profile_urn(value)
    except LinkedInScraperException:
        return None
    return candidate.removeprefix(_PROFILE_URN_PREFIX)


def _profile_path_from_url(value: str) -> str | None:
    parsed = _safe_linkedin_url(value)
    if parsed is None or parsed.query or not _PROFILE_PATH_RE.fullmatch(parsed.path):
        return None
    try:
        username = normalize_person_identifier(value)
    except LinkedInScraperException:
        return None
    canonical_path = urlparse(person_profile_url(username, "/")).path
    return parsed.path if parsed.path == canonical_path else None


def _profile_urn_from_compose_url(value: str, *, base: str | None = None) -> str | None:
    parsed = _safe_linkedin_url(value, base=base)
    if parsed is None or parsed.path != "/messaging/compose/":
        return None
    params = parse_qs(parsed.query, keep_blank_values=True)
    identifiers: set[str] = set()
    for key in ("recipient", "profileUrn"):
        values = params.get(key, [])
        normalized = [_normalize_profile_urn(item) for item in values]
        if any(item is None for item in normalized):
            return None
        identifiers.update(item for item in normalized if item is not None)
    if len(identifiers) != 1:
        return None
    return identifiers.pop()


def _enter_to_send_result(url: str) -> dict[str, Any]:
    """Report LinkedIn's "Press Enter to Send" preference as a user fix."""
    return contracts.message_action_result(
        url,
        "enter_to_send_enabled",
        "LinkedIn is set to 'Press Enter to Send', which hides the Send "
        "button this tool clicks. In LinkedIn Messaging, open the '...' menu "
        "next to 'Press Enter to Send', choose 'Click Send to send', then "
        "retry. Nothing was sent.",
        recipient_selected=True,
    )


def _message_page_url_is_safe(value: str, profile_urn: str) -> bool:
    parsed = _safe_linkedin_url(value)
    if parsed is None:
        return False

    params = parse_qs(parsed.query, keep_blank_values=True)
    recipient_values = [
        item for key in ("recipient", "profileUrn") for item in params.get(key, [])
    ]
    if parsed.path != "/messaging/compose/" and not _MESSAGE_THREAD_PATH_RE.fullmatch(
        parsed.path
    ):
        return False
    return all(_normalize_profile_urn(item) == profile_urn for item in recipient_values)


def _thread_page_url_is_safe(value: str, thread_path: str) -> bool:
    """Whether a page URL is exactly the requested thread and names nobody.

    A ``recipient`` or ``profileUrn`` on a thread route would be a person this
    target cannot corroborate, so any, even blank, refuses the route.
    """
    parsed = _safe_linkedin_url(value)
    if (
        parsed is None
        or parsed.path != thread_path
        or not _MESSAGE_THREAD_PATH_RE.fullmatch(parsed.path)
    ):
        return False
    params = parse_qs(parsed.query, keep_blank_values=True)
    return not any(key in params for key in ("recipient", "profileUrn"))


def _route_is_safe(value: str, target: _MessageTarget) -> bool:
    if isinstance(target, _ThreadMessageTarget):
        return _thread_page_url_is_safe(value, target.thread_path)
    return _message_page_url_is_safe(value, target.profile_urn)


class MessageSender:
    """Compose and send messages through LinkedIn's browser UI."""

    def __init__(self, session: ScrapingSession, navigator: PageNavigator):
        self._session = session
        self._navigator = navigator
        self._page = session.page

    async def _read_profile_message_target(self) -> _ProfileMessageTargetResolution:
        """Resolve one recipient-specific top-card compose action after settling."""
        try:
            await self._page.wait_for_function(
                _PROFILE_MESSAGE_TARGET_READY_JS,
                timeout=_PROFILE_MESSAGE_TARGET_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError:
            pass
        except Exception:
            logger.debug("Could not wait for the profile Message action", exc_info=True)

        try:
            data = await self._page.evaluate(_PROFILE_MESSAGE_TARGET_JS)
        except Exception:
            logger.debug("Could not inspect the profile Message action", exc_info=True)
            return _ProfileMessageTargetResolution("failed")
        if not isinstance(data, dict):
            return _ProfileMessageTargetResolution("failed")
        if data.get("status") == "unavailable":
            page_url = data.get("pageUrl")
            if (
                not isinstance(page_url, str)
                or _profile_path_from_url(page_url) is None
            ):
                return _ProfileMessageTargetResolution("failed")
            return _ProfileMessageTargetResolution("unavailable")
        if data.get("status") != "resolved":
            return _ProfileMessageTargetResolution("failed")

        page_url = data.get("pageUrl")
        compose_hrefs = data.get("composeHrefs")
        if not isinstance(page_url, str) or not isinstance(compose_hrefs, list):
            return _ProfileMessageTargetResolution("failed")
        profile_path = _profile_path_from_url(page_url)
        if profile_path is None:
            return _ProfileMessageTargetResolution("failed")
        if len(compose_hrefs) != 1 or not isinstance(compose_hrefs[0], str):
            return _ProfileMessageTargetResolution("failed")

        parsed_compose = _safe_linkedin_url(compose_hrefs[0], base=page_url)
        if parsed_compose is None:
            return _ProfileMessageTargetResolution("failed")
        compose_url = parsed_compose.geturl()
        profile_urn = _profile_urn_from_compose_url(compose_url)
        if profile_urn is None:
            return _ProfileMessageTargetResolution("failed")

        display_name = data.get("displayName")
        if not isinstance(display_name, str) or not display_name.strip():
            display_name = None
        else:
            display_name = display_name.strip()
        return _ProfileMessageTargetResolution(
            "resolved",
            _ProfileMessageTarget(
                profile_path=profile_path,
                profile_urn=profile_urn,
                compose_url=compose_url,
                display_name=display_name,
            ),
        )

    async def _resolve_message_compose_href(self) -> str | None:
        """Return an unambiguous recipient-specific top-card compose URL."""
        resolution = await self._read_profile_message_target()
        return resolution.target.compose_url if resolution.target else None

    async def _wait_for_message_surface(
        self, target: _MessageTarget
    ) -> Literal["composer"] | None:
        """Wait for one editor with no contradictory local recipient identity."""
        if await self._wait_for_message_composer(target):
            return "composer"
        return None

    async def _wait_for_message_composer(self, target: _MessageTarget) -> bool:
        """Wait for the complete verified LinkedIn composer state to settle.

        A thread target is ready only once its pane shows a message, so this
        is also the wait for the thread's history to hydrate, and it gets the
        longer ceiling a heavy thread needs.
        """
        try:
            if isinstance(target, _ThreadMessageTarget):
                await self._page.wait_for_function(
                    _MESSAGE_COMPOSER_READY_JS,
                    arg=self._message_target_argument(target),
                    timeout=_THREAD_READY_TIMEOUT_MS,
                )
            else:
                await self._page.wait_for_function(
                    _MESSAGE_COMPOSER_READY_JS,
                    arg=self._message_target_argument(target),
                )
        except PlaywrightTimeoutError:
            return False
        except Exception:
            logger.debug("Could not wait for the message editor", exc_info=True)
            return False
        return True

    async def _resolve_message_compose_box(self) -> Any | None:
        """Resolve the editor only when exactly one visible candidate exists."""
        locator = self._page.locator(f"{_MESSAGING_COMPOSE_SELECTOR}:visible")
        try:
            if await locator.count() != 1:
                return None
        except Exception:
            logger.debug("Could not count message editor candidates", exc_info=True)
            return None
        return locator.first

    @staticmethod
    def _message_target_argument(
        target: _MessageTarget,
    ) -> dict[str, str | bool]:
        # Exactly one shape per target. The page JS switches on the presence
        # of threadPath, so a thread argument must carry no profile fields
        # and a profile argument no threadPath.
        if isinstance(target, _ThreadMessageTarget):
            return {"threadPath": target.thread_path}
        return {
            "profilePath": target.profile_path,
            "profileUrn": target.profile_urn,
        }

    async def _read_message_composer_state(
        self, target: _MessageTarget
    ) -> dict[str, Any]:
        """Inspect the unique editor and reject contradictory local identity."""
        state = await self._page.evaluate(
            _MESSAGE_COMPOSER_STATE_JS,
            self._message_target_argument(target),
        )
        return state if isinstance(state, dict) else {"status": "invalid"}

    async def _focus_verified_message_editor(self, target: _MessageTarget) -> bool:
        """Focus the same editor after local contradiction checks."""
        focused = await self._page.evaluate(
            _MESSAGE_COMPOSER_FOCUS_JS,
            self._message_target_argument(target),
        )
        return focused is True

    async def _write_verified_message(
        self,
        message: str,
        *,
        target: _MessageTarget,
        owner: Any,
    ) -> str:
        """Insert text synchronously into the pinned local editor."""
        result = await owner.evaluate(
            _MESSAGE_COMPOSER_WRITE_JS,
            {**self._message_target_argument(target), "message": message},
        )
        return result if result in {"written", "occupied", "unsupported"} else "invalid"

    async def _wait_for_verified_submit(
        self,
        message: str,
        *,
        target: _MessageTarget,
        owner: Any,
    ) -> bool:
        """Wait briefly for the exact pinned submit button to become active."""
        timeout_ms = (
            _THREAD_SUBMIT_READY_TIMEOUT_MS
            if isinstance(target, _ThreadMessageTarget)
            else _MESSAGE_SUBMIT_READY_TIMEOUT_MS
        )
        deadline = time.monotonic() + timeout_ms / 1_000
        argument = {**self._message_target_argument(target), "message": message}
        while True:
            try:
                state = await owner.evaluate(
                    _MESSAGE_COMPOSER_SUBMIT_READY_JS, argument
                )
                if state == "ready":
                    return True
                if state != "disabled":
                    return False
            except Exception:
                logger.debug(
                    "Could not wait for the pinned submit button", exc_info=True
                )
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.05, remaining))

    async def _submit_verified_message(
        self,
        message: str,
        *,
        target: _MessageTarget,
        owner: Any,
    ) -> str:
        """Click the one active submit button pinned with the local editor."""
        result = await owner.evaluate(
            _MESSAGE_COMPOSER_SUBMIT_JS,
            {**self._message_target_argument(target), "message": message},
        )
        return "clicked" if result == "clicked" else "invalid"

    @staticmethod
    async def _cleanup_owned_message(message: str, owner: Any) -> None:
        """Best-effort removal of text proven to belong to this tool call."""
        with anyio.move_on_after(
            _MESSAGE_CLEANUP_TIMEOUT_SECONDS, shield=True
        ) as scope:
            try:
                await owner.evaluate(_MESSAGE_COMPOSER_CLEANUP_JS, {"message": message})
            except Exception:
                logger.debug("Could not clear tool-owned message text", exc_info=True)
        if scope.cancel_called:
            logger.warning("Timed out clearing tool-owned message text")
        await anyio.lowlevel.checkpoint()

    async def _resolve_message_owner(
        self,
        target: _MessageTarget,
        *,
        expected_route: str,
    ) -> Any | None:
        """Hold the verified owner node across submission and confirmation."""
        owner = await self._page.evaluate_handle(
            _MESSAGE_COMPOSER_OWNER_JS,
            arg={
                "target": self._message_target_argument(target),
                "expectedRoute": expected_route,
            },
        )
        if owner.as_element() is None:
            await self._dispose_message_owner(owner)
            return None
        return owner

    @staticmethod
    async def _dispose_message_owner(owner: Any) -> None:
        """Release all owner-scoped observers, pins, markers and handles."""
        try:
            with anyio.move_on_after(
                _MESSAGE_CLEANUP_TIMEOUT_SECONDS, shield=True
            ) as dom_scope:
                try:
                    await owner.evaluate(_MESSAGE_COMPOSER_DISPOSE_JS)
                except Exception:
                    logger.debug("Could not clear pinned message nodes", exc_info=True)
            if dom_scope.cancel_called:
                logger.warning("Timed out clearing pinned message nodes")
        finally:
            with anyio.move_on_after(
                _MESSAGE_CLEANUP_TIMEOUT_SECONDS, shield=True
            ) as handle_scope:
                try:
                    await owner.dispose()
                except Exception:
                    logger.debug(
                        "Could not release message owner handle", exc_info=True
                    )
            if handle_scope.cancel_called:
                logger.warning("Timed out releasing message owner handle")
        await anyio.lowlevel.checkpoint()

    def _message_confirmation_argument(
        self,
        message: str,
        target: _MessageTarget,
        owner: Any,
    ) -> dict[str, Any]:
        return {
            **self._message_target_argument(target),
            "expected": message,
            "owner": owner,
        }

    async def _prepare_message_confirmation(
        self,
        message: str,
        *,
        target: _MessageTarget,
        owner: Any,
    ) -> str | None:
        """Start the owner-scoped DOM observer immediately before submission."""
        token = await self._page.evaluate(
            _MESSAGE_CONFIRMATION_PREPARE_JS,
            self._message_confirmation_argument(message, target, owner),
        )
        return token if isinstance(token, str) and token else None

    async def _message_send_confirmed(
        self,
        message: str,
        *,
        target: _MessageTarget,
        owner: Any,
        confirmation: str,
    ) -> bool:
        """Wait for LinkedIn to acknowledge the submitted message in its thread.

        Two signals count. The observer accepts a node inserted after it was
        installed whose exact visible message unit equals the typed text and
        which then changes from one non-empty ``data-event-urn`` value to a
        different one. Otherwise the conversation pane of the one composer on
        the page must hold exactly one visible exact-text node whose event ID
        is a server message URN absent before submission: LinkedIn replaces
        its client placeholder with that node, and the first message of a new
        thread remounts the pane under /messaging/thread/<id>/. Every timeout
        or ambiguity answers "not observed" because submission already
        happened.

        A thread target narrows the second signal: the route never left the
        named thread, the observer saw LinkedIn's client placeholder for this
        text leave the pane, and the node's sender is nobody the pane header
        names. It also waits longer, since a heavy thread re-renders slowly.
        """
        argument = {
            **self._message_target_argument(target),
            "expected": message,
            "owner": owner,
            "token": confirmation,
        }
        try:
            if isinstance(target, _ThreadMessageTarget):
                await self._page.wait_for_function(
                    _MESSAGE_CONFIRMATION_READY_JS,
                    arg=argument,
                    timeout=_THREAD_CONFIRMATION_TIMEOUT_MS,
                )
            else:
                await self._page.wait_for_function(
                    _MESSAGE_CONFIRMATION_READY_JS, arg=argument
                )
            return True
        except Exception:
            logger.debug("Message send could not be confirmed", exc_info=True)
            return False

    async def _dispose_message_confirmation(
        self, owner: Any, confirmation: str
    ) -> None:
        """Disconnect a request-local confirmation observer."""
        with anyio.move_on_after(
            _MESSAGE_CLEANUP_TIMEOUT_SECONDS, shield=True
        ) as scope:
            try:
                await self._page.evaluate(
                    _MESSAGE_CONFIRMATION_DISPOSE_JS,
                    {"owner": owner, "token": confirmation},
                )
            except Exception:
                logger.debug("Could not disconnect message observer", exc_info=True)
        if scope.cancel_called:
            logger.warning("Timed out disconnecting message observer")
        await anyio.lowlevel.checkpoint()

    async def send_message(
        self,
        linkedin_username: str,
        message: str,
        *,
        confirm_send: bool,
        profile_urn: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Compose and send a message with explicit confirmation gating.

        Without ``thread_id`` this opens LinkedIn's profile-based compose flow.
        That may create a separate DM instead of replying in an existing
        recruiter/InMail or messaging thread. Recipient authorization comes from
        the validated top-card action carrying the target URN and the browser
        navigation it initiates. The exact resulting route is pinned through
        every later operation; visible local identities are optional
        corroboration, but any contradiction fails closed.

        With ``thread_id`` it replies inside that existing thread instead
        (#483), and never falls back to the profile flow: see
        :meth:`_send_in_thread`. ``linkedin_username`` is then ignored, neither
        validated nor used, and ``profile_urn`` is refused because a thread
        page offers nothing to verify it against.

        Args:
            linkedin_username: LinkedIn username of the recipient. Ignored when
                ``thread_id`` is given.
            message: The message text to send.
            confirm_send: Must be True to actually send (False does a dry run).
            profile_urn: Optional profile URN (e.g. ACoAAB...) to verify against
                the recipient resolved from the loaded profile snapshot.
            thread_id: Optional id of an existing conversation to reply in.
        """
        if thread_id is not None:
            if profile_urn is not None:
                raise InvalidReferenceError(contracts.THREAD_REPLY_PROFILE_URN_REFUSAL)
            return await self._send_in_thread(
                thread_id, message, confirm_send=confirm_send
            )

        refusal = contracts.refuse_an_invalid_message(linkedin_username, message)
        if refusal is not None:
            return refusal
        linkedin_username = normalize_person_identifier(linkedin_username)
        if profile_urn is not None:
            profile_urn = normalize_profile_urn(profile_urn)
        profile_url = person_profile_url(linkedin_username, "/")

        await self._navigator._navigate_to_page(profile_url)
        await self._session.check_rate_limit()

        try:
            await self._page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("Profile page did not load for %s", linkedin_username)

        resolution = await self._read_profile_message_target()
        if resolution.status == "unavailable":
            return contracts.message_action_result(
                profile_url,
                "message_unavailable",
                "LinkedIn did not expose a normal Message action for this profile. "
                "Use connect_with_person first, then retry only after the connection "
                "request is accepted.",
            )
        target = resolution.target
        if target is None:
            return contracts.message_action_result(
                profile_url,
                "recipient_resolution_failed",
                "LinkedIn did not expose one unambiguous recipient-specific Message "
                "action.",
            )

        supplied_urn = _normalize_profile_urn(profile_urn) if profile_urn else None
        if profile_urn is not None and supplied_urn != target.profile_urn:
            return contracts.message_action_result(
                profile_url,
                "recipient_resolution_failed",
                "The supplied profile URN did not match the loaded profile.",
            )

        # The validated top-card action and its browser navigation are the
        # recipient boundary. LinkedIn may strip the query and expose no local
        # identity, so capture the final route now and fail on any later change or
        # visible contradiction. Do not replace this with a Voyager/private API.
        # See docs/decisions/2026-09-16-rendered-page.md.
        await self._navigator._navigate_to_page(target.compose_url)
        expected_route = self._page.url
        if not _message_page_url_is_safe(expected_route, target.profile_urn):
            return contracts.message_action_result(
                expected_route,
                "recipient_resolution_failed",
                "LinkedIn opened an unexpected messaging URL.",
            )
        return await self._send_on_pinned_route(
            target,
            message,
            expected_route=expected_route,
            confirm_send=confirm_send,
            label=linkedin_username,
        )

    async def _send_in_thread(
        self,
        thread_id: str,
        message: str,
        *,
        confirm_send: bool,
    ) -> dict[str, Any]:
        """Reply inside the existing thread ``thread_id`` names (#483).

        The thread id is the whole target. It is validated into the exact
        ``/messaging/thread/<id>/`` route before any browser work, the page is
        opened there, and the route it lands on must be that one: LinkedIn
        redirecting an unknown or foreign thread elsewhere is
        ``thread_unavailable``, never a reason to try the profile flow. From
        then on the route is pinned exactly like a profile send's, and the
        composer has to be the one docked in that thread's own pane. The rest
        of the path is the profile send's, step for step.
        """
        target = _thread_message_target(thread_id)
        refusal = contracts.refuse_an_invalid_thread_message(target.thread_id, message)
        if refusal is not None:
            return refusal

        await self._navigator._navigate_to_page(target.thread_url)
        # Before the landing check, so a checkpoint reached instead of the
        # thread reports itself as one rather than as a missing thread.
        await self._session.check_rate_limit()
        expected_route = self._page.url
        if not _thread_page_url_is_safe(expected_route, target.thread_path):
            return contracts.message_action_result(
                expected_route,
                "thread_unavailable",
                "LinkedIn did not open the requested messaging thread, so nothing "
                "was typed or sent. Check the thread_id: pass it exactly as "
                "get_inbox, get_conversation or search_conversations returned it.",
            )
        return await self._send_on_pinned_route(
            target,
            message,
            expected_route=expected_route,
            confirm_send=confirm_send,
            label=f"thread {target.thread_id}",
        )

    async def _wait_for_main(self, target: _MessageTarget, label: str) -> None:
        """Let the messaging page's ``main`` render; its absence is not fatal."""
        try:
            if isinstance(target, _ThreadMessageTarget):
                await self._page.wait_for_selector(
                    "main", timeout=_THREAD_READY_TIMEOUT_MS
                )
            else:
                await self._page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("Compose page did not fully load for %s", label)

    async def _send_on_pinned_route(
        self,
        target: _MessageTarget,
        message: str,
        *,
        expected_route: str,
        confirm_send: bool,
        label: str,
    ) -> dict[str, Any]:
        """Verify, write, submit and confirm on the route captured on arrival.

        Shared by both targets from the moment the route is pinned, so a thread
        reply clears every gate a profile send does: the unchanged route after
        each await, one verified composer, the dry run, an occupied draft,
        Enter-to-send, one pinned submit, the verified write, the confirmation
        observer, and cleanup of text nothing submitted.
        """
        await self._session.check_rate_limit()
        if self._page.url != expected_route:
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The messaging URL changed while the composer was loading.",
            )

        await self._wait_for_main(target, label)
        if self._page.url != expected_route:
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The messaging URL changed while the composer was loading.",
            )

        message_surface = await self._wait_for_message_surface(target)
        if self._page.url != expected_route:
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The messaging URL changed while the composer was loading.",
            )
        logger.debug("Message surface for %s was %s", label, message_surface)
        if message_surface != "composer":
            return contracts.message_action_result(
                self._page.url,
                "composer_unavailable",
                "LinkedIn did not expose one usable message composer.",
            )

        state = await self._read_message_composer_state(target)
        if self._page.url != expected_route:
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The messaging URL changed during recipient verification.",
            )
        if state.get("status") != "valid":
            logger.debug(
                "Message recipient verification for %s returned %s",
                label,
                state.get("status"),
            )
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The local composer did not identify exactly the requested thread."
                if isinstance(target, _ThreadMessageTarget)
                else "The local composer did not identify exactly the requested "
                "profile.",
            )
        recipient_selected = True
        if state.get("enterToSend") is True:
            return _enter_to_send_result(self._page.url)

        if not confirm_send:
            return contracts.message_action_result(
                self._page.url,
                "confirmation_required",
                "Set confirm_send=true to send the message.",
                recipient_selected=recipient_selected,
            )

        if self._page.url != expected_route:
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The messaging URL changed before text entry.",
                recipient_selected=recipient_selected,
            )
        state = await self._read_message_composer_state(target)
        if self._page.url != expected_route:
            return contracts.message_action_result(
                self._page.url,
                "recipient_resolution_failed",
                "The messaging URL changed before text entry.",
                recipient_selected=recipient_selected,
            )
        if state.get("status") == "valid" and state.get("empty") is not True:
            # Text already in the editor belongs to whoever typed it. Clearing
            # it would trade a recipient leak for destroying their draft.
            return contracts.message_action_result(
                self._page.url,
                "composer_occupied",
                "The composer already holds a draft that would be sent along "
                "with the message. The draft was left untouched.",
                recipient_selected=recipient_selected,
            )
        if state.get("status") != "valid":
            return contracts.message_action_result(
                self._page.url,
                "compose_interact_failed",
                "The verified message composer changed before text entry.",
                recipient_selected=recipient_selected,
            )
        if state.get("enterToSend") is True:
            return _enter_to_send_result(self._page.url)
        if state.get("submitCount") != 1:
            return contracts.message_action_result(
                self._page.url,
                "send_unavailable",
                "The local submit path was missing or ambiguous.",
                recipient_selected=recipient_selected,
            )

        may_have_submitted = False
        try:
            owner = await self._resolve_message_owner(
                target, expected_route=expected_route
            )
            if owner is None:
                return contracts.message_action_result(
                    self._page.url,
                    "recipient_resolution_failed",
                    "The verified message composer changed before text entry.",
                    recipient_selected=recipient_selected,
                )

            try:
                write_result = await self._write_verified_message(
                    message,
                    target=target,
                    owner=owner,
                )
                if not _route_is_safe(self._page.url, target):
                    return contracts.message_action_result(
                        self._page.url,
                        "recipient_resolution_failed",
                        "The messaging URL changed during text entry.",
                        recipient_selected=recipient_selected,
                    )
                if write_result == "occupied":
                    return contracts.message_action_result(
                        self._page.url,
                        "composer_occupied",
                        "The composer already holds a draft that would be sent along "
                        "with the message. The draft was left untouched.",
                        recipient_selected=recipient_selected,
                    )
                if write_result != "written":
                    return contracts.message_action_result(
                        self._page.url,
                        "compose_interact_failed",
                        "The verified message editor could not accept the message.",
                        recipient_selected=recipient_selected,
                    )

                if not await self._wait_for_verified_submit(
                    message,
                    target=target,
                    owner=owner,
                ):
                    return contracts.message_action_result(
                        self._page.url,
                        "send_unavailable",
                        "The pinned submit button did not become available without "
                        "changing the verified composer.",
                        recipient_selected=recipient_selected,
                    )

                confirmation = await self._prepare_message_confirmation(
                    message,
                    target=target,
                    owner=owner,
                )
                if confirmation is None:
                    return contracts.message_action_result(
                        self._page.url,
                        "recipient_resolution_failed",
                        "The verified message composer changed before submission.",
                        recipient_selected=recipient_selected,
                    )

                try:
                    try:
                        # A click can dispatch before the evaluate call reports an
                        # error, so an exception from this round trip is ambiguous.
                        may_have_submitted = True
                        submission = await self._submit_verified_message(
                            message,
                            target=target,
                            owner=owner,
                        )
                    except Exception:
                        logger.debug(
                            "Message submission did not complete", exc_info=True
                        )
                        return contracts.message_action_result(
                            self._page.url,
                            "send_unconfirmed",
                            "The message submission was interrupted and LinkedIn did "
                            "not confirm the send. Check the conversation before "
                            "retrying; retrying may deliver the message twice.",
                            recipient_selected=recipient_selected,
                            retry_safe=False,
                        )

                    if submission != "clicked":
                        may_have_submitted = False
                        return contracts.message_action_result(
                            self._page.url,
                            "send_unavailable",
                            "The local submit path was missing, disabled, or ambiguous.",
                            recipient_selected=recipient_selected,
                        )

                    confirmed = await self._message_send_confirmed(
                        message,
                        target=target,
                        owner=owner,
                        confirmation=confirmation,
                    )
                    if not confirmed:
                        return contracts.message_action_result(
                            self._page.url,
                            "send_unconfirmed",
                            "The message was submitted but LinkedIn did not confirm "
                            "the message-list transition in time. Check the "
                            "conversation before retrying; retrying may deliver the "
                            "message twice.",
                            recipient_selected=recipient_selected,
                            retry_safe=False,
                        )

                    return contracts.message_action_result(
                        self._page.url,
                        "sent",
                        "Message submitted and confirmed in the conversation UI.",
                        recipient_selected=recipient_selected,
                        sent=True,
                        retry_safe=False,
                    )
                finally:
                    await self._dispose_message_confirmation(owner, confirmation)
            finally:
                try:
                    if not may_have_submitted:
                        await self._cleanup_owned_message(message, owner)
                finally:
                    await self._dispose_message_owner(owner)
        except Exception:
            if not may_have_submitted:
                # Nothing can have been submitted yet, so the error itself is
                # the useful answer and the caller can retry on it.
                raise
            logger.debug(
                "Message send failed after a possible submission", exc_info=True
            )
            return contracts.message_action_result(
                self._page.url,
                "send_unconfirmed",
                "The message may already have been submitted when the send "
                "failed, and LinkedIn did not confirm the outcome. Check the "
                "conversation before retrying; retrying may deliver the "
                "message twice.",
                recipient_selected=recipient_selected,
                retry_safe=False,
            )
        except BaseException:
            # Cancellation only. FastMCP runs the tool inside
            # `anyio.fail_after()` and a cancelled scope discards whatever it
            # returns, so the answer the branch above gives cannot be given
            # here and the log line is all that is left.
            #
            # Silent before explicit submission: nothing can have left yet,
            # and a warning about duplicate delivery would be false.
            if may_have_submitted:
                logger.warning(contracts.SEND_INTERRUPTED_WARNING)
            raise
