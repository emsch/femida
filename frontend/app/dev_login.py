"""DEV-ONLY login bypass for the local stand. Inert unless FEMIDA_DEV_LOGIN is set.

Google OAuth cannot complete on localhost, so when FEMIDA_DEV_LOGIN holds an
e-mail the ``/login`` route signs that e-mail in directly and the user loader
stops asking Google. Flask-OAuthlib still needs non-empty credentials to
import, so the stand sets both GOOGLE_ID and GOOGLE_SECRET to ``PLACEHOLDER``.
The bypass refuses to start with any other value: a production envfile carries
a real client id, so enabling it there would take two deliberate edits.
"""
import logging
import os

logger = logging.getLogger(__name__)

DEV_LOGIN_EMAIL = os.environ.get('FEMIDA_DEV_LOGIN', '')
PLACEHOLDER = 'dev-login-bypass'


def enabled():
    return bool(DEV_LOGIN_EMAIL)


def refuse_production():
    if not enabled():
        return
    credentials = (os.environ.get('GOOGLE_ID'), os.environ.get('GOOGLE_SECRET'))
    if credentials != (PLACEHOLDER, PLACEHOLDER):
        raise RuntimeError(
            'FEMIDA_DEV_LOGIN is set, but GOOGLE_ID/GOOGLE_SECRET are not the '
            f'"{PLACEHOLDER}" placeholder. The dev login bypass must never run '
            'next to real OAuth credentials: unset FEMIDA_DEV_LOGIN.'
        )
    logger.warning('DEV LOGIN BYPASS ACTIVE: /login signs in %s without Google',
                   DEV_LOGIN_EMAIL)


def make_user(user_cls):
    return user_cls('dev', DEV_LOGIN_EMAIL, '')


def load_user(userid, user_cls):
    if userid != DEV_LOGIN_EMAIL:
        return None
    return make_user(user_cls)
