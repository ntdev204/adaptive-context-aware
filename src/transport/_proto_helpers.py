"""Shared protobuf descriptor helpers for the transport data plane."""

from __future__ import annotations

from google.protobuf import descriptor_pb2


def add_proto_field(
    message_desc: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    *,
    repeated: bool = False,
    type_name: str | None = None,
    oneof_index: int | None = None,
    proto3_optional: bool = False,
) -> None:
    """Append a field descriptor to *message_desc*."""
    field_desc = message_desc.field.add()
    field_desc.name = name
    field_desc.number = number
    field_desc.label = (
        descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        if repeated
        else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    )
    field_desc.type = field_type
    if type_name is not None:
        field_desc.type_name = type_name
    if oneof_index is not None:
        field_desc.oneof_index = oneof_index
    if proto3_optional:
        oneof_decl = message_desc.oneof_decl.add()
        oneof_decl.name = f"_{name}"
        field_desc.oneof_index = len(message_desc.oneof_decl) - 1
        field_desc.proto3_optional = True
