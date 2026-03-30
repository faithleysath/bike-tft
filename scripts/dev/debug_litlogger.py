#!/usr/bin/env python3
"""Diagnose Lightning.ai / litlogger authentication and experiment creation issues."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightning_sdk.lightning_cloud import env as lightning_env
from lightning_sdk.lightning_cloud.login import Auth
from lightning_sdk.lightning_cloud.openapi import LitLoggerServiceCreateMetricsStreamBody, V1SystemInfo
from lightning_sdk.lightning_cloud.openapi.rest import ApiException
from lightning_sdk.lightning_cloud.rest_client import GridRestClient, create_swagger_client
from litlogger.api.metrics_api import MetricsApi
from litlogger.colors import _create_colors
from litlogger.diagnostics import collect_system_info


def mask_secret(value: str | None, keep: int = 4) -> str:
    """Mask a credential while leaving enough visible to distinguish it."""
    if not value:
        return "<missing>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


def print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


def print_kv(key: str, value: Any) -> None:
    print(f"{key}: {value}")


def describe_api_exception(exc: ApiException) -> None:
    print_kv("status", getattr(exc, "status", None))
    print_kv("reason", getattr(exc, "reason", None))
    body = getattr(exc, "body", None)
    if body:
        print("body:")
        print(body)
    headers = getattr(exc, "headers", None)
    if headers:
        print("headers:")
        print(json.dumps(dict(headers), indent=2, sort_keys=True))


def load_auth_without_browser() -> tuple[Auth, str]:
    """Load credentials from env or disk without triggering browser auth."""
    auth = Auth()
    if auth._with_env_var:
        return auth, "environment"
    if auth.load():
        return auth, "credentials_file"
    return auth, "missing"


def build_client(auth: Auth) -> GridRestClient:
    """Create a REST client using already-loaded credentials only."""
    api_client = create_swagger_client(with_auth=False)
    auth_header = auth.get_auth_header()
    if not auth_header:
        raise RuntimeError("No cached Lightning credentials available. Cannot build authenticated client.")
    api_client.default_headers["Authorization"] = auth_header
    return GridRestClient(api_client=api_client)


def membership_summary(membership: Any) -> str:
    project_id = getattr(membership, "project_id", None)
    name = getattr(membership, "name", None)
    display_name = getattr(membership, "display_name", None)
    owner_id = getattr(membership, "owner_id", None)
    owner_type = getattr(membership, "owner_type", None)
    return (
        f"id={project_id} name={name} display_name={display_name} "
        f"owner_type={owner_type} owner_id={owner_id}"
    )


def find_membership(memberships: list[Any], requested: str | None) -> Any | None:
    if not memberships:
        return None
    if requested is None:
        return memberships[0]

    for membership in memberships:
        if requested in {
            getattr(membership, "project_id", None),
            getattr(membership, "name", None),
            getattr(membership, "display_name", None),
        }:
            return membership
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose litlogger auth and metrics-stream creation issues.")
    parser.add_argument(
        "--teamspace",
        default=None,
        help="Teamspace identifier to test. Accepts project_id, name, or display_name. Defaults to litlogger's first membership behavior.",
    )
    parser.add_argument(
        "--experiment-name",
        default=f"litlogger-debug-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        help="Experiment name to probe with.",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Skip the create_metrics_stream call and only inspect auth/teamspace/list permissions.",
    )
    args = parser.parse_args()

    print_section("Environment")
    print_kv("python", sys.version.split()[0])
    print_kv("hostname", socket.gethostname())
    print_kv("cwd", Path.cwd())
    print_kv("LIGHTNING_CLOUD_URL", lightning_env.LIGHTNING_CLOUD_URL)
    print_kv("LIGHTNING_CREDENTIAL_PATH", lightning_env.LIGHTNING_CREDENTIAL_PATH)
    print_kv("LIGHTNING_CLOUD_PROJECT_ID", os.getenv("LIGHTNING_CLOUD_PROJECT_ID"))
    print_kv("LIGHTNING_CLOUD_SPACE_ID", os.getenv("LIGHTNING_CLOUD_SPACE_ID"))
    print_kv("LIGHTNING_CLOUD_APP_ID", os.getenv("LIGHTNING_CLOUD_APP_ID"))
    print_kv("LIGHTNING_CLOUD_WORK_ID", os.getenv("LIGHTNING_CLOUD_WORK_ID"))

    print_section("Credential Summary")
    auth, source = load_auth_without_browser()
    cred_path = Path(lightning_env.LIGHTNING_CREDENTIAL_PATH)
    print_kv("credential_source", source)
    print_kv("credentials_file_exists", cred_path.exists())
    print_kv("user_id", mask_secret(auth.user_id))
    print_kv("api_key", mask_secret(auth.api_key))
    print_kv("auth_token", mask_secret(auth.auth_token))
    if source == "missing":
        print("No cached Lightning credentials were found. This script will not attempt browser login.")
        print("If you want to test guest mode, run the training script directly or copy a valid credentials file to the server.")
        return 2

    try:
        client = build_client(auth)
    except Exception as exc:
        print_section("Client Build Failed")
        print(repr(exc))
        return 2

    print_section("Current User")
    try:
        user = client.auth_service_get_user()
        print_kv("id", getattr(user, "id", None))
        print_kv("username", getattr(user, "username", None))
        print_kv("email", getattr(user, "email", None))
    except ApiException as exc:
        print("auth_service_get_user failed")
        describe_api_exception(exc)
        return 1

    print_section("Memberships")
    try:
        memberships_response = client.projects_service_list_memberships()
    except ApiException as exc:
        print("projects_service_list_memberships failed")
        describe_api_exception(exc)
        return 1

    memberships = list(getattr(memberships_response, "memberships", []) or [])
    print_kv("membership_count", len(memberships))
    for index, membership in enumerate(memberships):
        print(f"[{index}] {membership_summary(membership)}")

    selected = find_membership(memberships, args.teamspace)
    if selected is None:
        print_section("Teamspace Resolution")
        if args.teamspace is None:
            print("No memberships available, so litlogger has no teamspace to log into.")
        else:
            print(f"Requested teamspace {args.teamspace!r} was not found in the available memberships above.")
        return 1

    selected_project_id = getattr(selected, "project_id", None)
    selected_name = getattr(selected, "name", None)
    if not isinstance(selected_project_id, str) or not selected_project_id:
        print_section("Selected Teamspace")
        print("The selected membership does not expose a valid project_id, so litlogger cannot target it.")
        return 1

    print_section("Selected Teamspace")
    print_kv("requested_teamspace", args.teamspace or "<default first membership>")
    print_kv("selected_project_id", selected_project_id)
    print_kv("selected_name", selected_name)
    print_kv("selected_display_name", getattr(selected, "display_name", None))
    if args.teamspace is None and memberships:
        print("This matches litlogger's default behavior when no --litlogger-teamspace is provided.")

    print_section("List Existing Metrics Streams")
    try:
        streams_response = client.lit_logger_service_list_metrics_streams(project_id=selected_project_id)
    except ApiException as exc:
        print("lit_logger_service_list_metrics_streams failed")
        describe_api_exception(exc)
        return 1

    streams = list(getattr(streams_response, "metrics_streams", []) or [])
    print_kv("stream_count", len(streams))
    exact_match = None
    for stream in streams:
        if getattr(stream, "name", None) == args.experiment_name:
            exact_match = stream
            break
    if exact_match is None:
        print_kv("existing_stream_for_requested_name", "<none>")
    else:
        print_kv("existing_stream_for_requested_name", getattr(exact_match, "id", None))

    if args.no_create:
        print_section("Done")
        print("Skipped create_metrics_stream because --no-create was provided.")
        return 0

    print_section("Create Metrics Stream Request")
    metadata = {
        "debug_script": "scripts/dev/debug_litlogger.py",
        "host": socket.gethostname(),
    }
    light_color, dark_color = _create_colors(args.experiment_name)
    cloudspace_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID")
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID")
    work_id = os.getenv("LIGHTNING_CLOUD_WORK_ID")
    same_project_as_context = selected_project_id == os.getenv("LIGHTNING_CLOUD_PROJECT_ID")
    if not same_project_as_context:
        cloudspace_id = None
        app_id = None
        work_id = None

    print_kv("experiment_name", args.experiment_name)
    print_kv("same_project_as_context", same_project_as_context)
    print_kv("cloudspace_id_sent", cloudspace_id)
    print_kv("app_id_sent", app_id)
    print_kv("work_id_sent", work_id)
    print_kv("metadata", metadata)

    body_kwargs: dict[str, Any] = {
        "name": args.experiment_name,
        "light_color": light_color,
        "dark_color": dark_color,
        "tags": MetricsApi._metadata_to_tags(metadata),
        "store_step": True,
        "store_created_at": True,
        "system_info": V1SystemInfo(**collect_system_info()),
    }
    if cloudspace_id is not None:
        body_kwargs["cloudspace_id"] = cloudspace_id
    if app_id is not None:
        body_kwargs["app_id"] = app_id
    if work_id is not None:
        body_kwargs["work_id"] = work_id
    body = LitLoggerServiceCreateMetricsStreamBody(**body_kwargs)

    try:
        created = client.lit_logger_service_create_metrics_stream(project_id=selected_project_id, body=body)
    except ApiException as exc:
        print("lit_logger_service_create_metrics_stream failed")
        describe_api_exception(exc)
        print()
        print("Likely next checks:")
        print("- Try --teamspace with a different membership from the list above.")
        print("- Verify the logged-in account actually has LitLogger access in that teamspace.")
        print("- If list_metrics_streams works but create_metrics_stream is 403, this is likely a write-permission issue.")
        return 1

    print_section("Create Metrics Stream Result")
    print_kv("created_stream_id", getattr(created, "id", None))
    print_kv("created_stream_name", getattr(created, "name", None))
    print_kv("created_stream_cloudspace_id", getattr(created, "cloudspace_id", None))
    print("Success: this account can create litlogger experiments in the selected teamspace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
