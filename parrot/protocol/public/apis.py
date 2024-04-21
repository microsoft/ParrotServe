# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List
import aiohttp

from parrot.utils import get_logger

from ..base_response import BaseResponse
from ..http_utils import async_send_http_request, send_http_request
from .api_version import API_VERSION


"""
Public APIs for users (served by ServeLayer).

Session (RESTful):
    - register_session (`/session/`, POST)
    - get_session_info (`/session/{session_id}`, GET)
    - remove_session POST (`/session/{session_id}`, DELETE)

Function Call:
    - submit_semantic_call POST
    - submit_native_call POST (TODO)

Semantic Variable (RESTful):
    - register_semantic_variable (`/semantic_var/`, POST)
    - set_semantic_variable (`/semantic_var/{var_id}`, POST)
    - get_semantic_variable (`/semantic_var/{var_id}`, GET)
    - get_semantic_variable_list (`/semantic_var/`, GET)
"""


logger = get_logger("Public API")


# ---------- Responses ----------


class RegisterSessionResponse(BaseResponse):
    session_id: int
    session_auth: str


class RemoveSessionResponse(BaseResponse):
    pass


class SubmitSemanticCallResponse(BaseResponse):
    request_id: int
    placeholders_mapping: List


class RegisterSemanticVariableResponse(BaseResponse):
    var_id: str


class SetSemanticVariableResponse(BaseResponse):
    pass


class GetSemanticVariableResponse(BaseResponse):
    content: str


class GetSemanticVariableListResponse(BaseResponse):
    pass


# ---------- APIs ----------


def register_session(http_addr: str, api_key: str) -> RegisterSessionResponse:
    try:
        return send_http_request(
            RegisterSessionResponse,
            http_addr,
            f"/{API_VERSION}/session",
            retry_times=1,
            api_key=api_key,
        )
    except BaseException as e:
        logger.error(f"Register session error in {http_addr}. Error: {e}")
        raise e


def get_session_info(http_addr: str, session_id: int, session_auth: str) -> Dict:
    try:
        return send_http_request(
            BaseResponse,
            http_addr,
            f"/{API_VERSION}/session/{session_id}",
            method="GET",
            retry_times=1,
            session_auth=session_auth,
        )
    except BaseException as e:
        logger.error(f"Get session info error in {http_addr}. Error: {e}")
        raise e


def remove_session(
    http_addr: str, session_id: int, session_auth: str
) -> RemoveSessionResponse:
    try:
        send_http_request(
            RemoveSessionResponse,
            http_addr,
            f"/{API_VERSION}/session/{session_id}",
            retry_times=1,
            session_auth=session_auth,
            method="DELETE",
        )
    except BaseException as e:
        logger.error(f"Remove session error in {http_addr}. Error: {e}")
        raise e


def submit_semantic_call(
    http_addr: str, session_id: int, session_auth: str, payload: Dict
) -> SubmitSemanticCallResponse:
    try:
        return send_http_request(
            SubmitSemanticCallResponse,
            http_addr,
            f"/{API_VERSION}/submit_semantic_call",
            retry_times=1,
            session_id=session_id,
            **payload,
        )
    except BaseException as e:
        logger.error(
            f"Submit call (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e


async def asubmit_semantic_call(
    http_addr: str, session_id: int, session_auth: str, payload: Dict
) -> SubmitSemanticCallResponse:
    try:
        async with aiohttp.ClientSession() as client_session:
            return await async_send_http_request(
                client_session,
                SubmitSemanticCallResponse,
                http_addr,
                f"/{API_VERSION}/submit_semantic_call",
                retry_times=1,
                session_id=session_id,
                **payload,
            )
    except BaseException as e:
        logger.error(
            f"Submit call (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e


def register_semantic_variable(
    http_addr: str,
    session_id: int,
    session_auth: str,
    var_name: str,
) -> RegisterSemanticVariableResponse:
    try:
        return send_http_request(
            RegisterSemanticVariableResponse,
            http_addr,
            f"/{API_VERSION}/semantic_var",
            retry_times=1,
            session_id=session_id,
            session_auth=session_auth,
            var_name=var_name,
        )
    except BaseException as e:
        logger.error(
            f"Register semantic variable (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e


def set_semantic_variable(
    http_addr: str, session_id: int, session_auth: str, var_id: str, content: str
) -> SetSemanticVariableResponse:
    try:
        return send_http_request(
            SetSemanticVariableResponse,
            http_addr,
            f"/{API_VERSION}/semantic_var/{var_id}",
            retry_times=1,
            session_id=session_id,
            session_auth=session_auth,
            content=content,
        )
    except BaseException as e:
        logger.error(
            f"Set semantic variable {var_id} (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e


def get_semantic_variable(
    http_addr: str, session_id: int, session_auth: str, var_id: str, criteria: str
) -> GetSemanticVariableResponse:
    try:
        return send_http_request(
            GetSemanticVariableResponse,
            http_addr,
            f"/{API_VERSION}/semantic_var/{var_id}",
            method="GET",
            retry_times=1,
            session_id=session_id,
            session_auth=session_auth,
            criteria=criteria,
        )
    except BaseException as e:
        logger.error(
            f"Get semantic variable {var_id} (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e


async def aget_semantic_variable(
    http_addr: str, session_id: int, session_auth: str, var_id: str, criteria: str
) -> GetSemanticVariableResponse:
    try:
        async with aiohttp.ClientSession() as client_session:
            return await async_send_http_request(
                client_session,
                GetSemanticVariableResponse,
                http_addr,
                f"/{API_VERSION}/semantic_var/{var_id}",
                method="GET",
                retry_times=1,
                session_id=session_id,
                session_auth=session_auth,
                criteria=criteria,
            )
    except BaseException as e:
        logger.error(
            f"AGet semantic variable {var_id} (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e


def get_semantic_variable_list(
    http_addr: str, session_id: int, session_auth: str
) -> GetSemanticVariableListResponse:
    try:
        return send_http_request(
            GetSemanticVariableListResponse,
            http_addr,
            f"/{API_VERSION}/semantic_var",
            method="GET",
            retry_times=1,
            session_id=session_id,
            session_auth=session_auth,
        )
    except BaseException as e:
        logger.error(
            f"Get semantic variable list (session_id={session_id}) error in {http_addr}. Error: {e}"
        )
        raise e
