# Wire Protocol Boundary

Portable JSON contracts and domain objects are separate concerns. Domain values
model valid runtime state; an implementation's wire boundary explicitly encodes
and decodes the portable documents defined in `contracts/v0`.

## Ownership

The kernel wire component owns codecs for portable runtime inputs, checkpoints,
events, and traces. A top-level codec composes private codecs for nested snapshots, state,
messages, model values, tool values, metrics, facts, and errors.

The portable start/continue/resume request union is a transport document. It
does not require a public domain request hierarchy: a host decodes its kind and
calls the corresponding `Runtime` method with typed arguments.

Domain classes do not expose generic `to_dict` or `from_dict` methods. Provider
HTTP formats remain in the provider layer and tool input/output JSON Schema
validation remains in the toolkit layer.

## Decode Rules

Every decoder:

1. selects the expected contract and verifies schema version when that
   top-level document is versioned;
2. rejects missing and unknown fields;
3. dispatches a union by its explicit discriminator;
4. validates scalar types without language-specific coercion;
5. rejects non-finite numbers and cyclic host input;
6. rejects JSON containers nested more than 128 levels;
7. validates cross-field invariants;
8. constructs frozen domain values once.

JSON boolean is not accepted where an integer or number is required. Empty
identifiers, duplicate tool-call ids, impossible state payloads, invalid
revision relationships, and mismatched checkpoint facts are protocol errors.

The interoperable integer, number, depth, and opaque-data acceptance rules are defined
once in [`contracts/v0`](../contracts/v0/README.md#boundary-rules).

Schema version is a property of the top-level wire document. It is not stored
on every domain event, snapshot, state, or value.

## Encode Rules

An encoder accepts only a valid domain aggregate and emits one canonical JSON
shape:

- mappings use contract field names exactly;
- union discriminators are explicit;
- omitted optional values follow the schema rather than becoming ad hoc nulls;
- immutable JSON objects are thawed only at this boundary;
- derived display values are emitted only when the contract owns them;
- provider-local metadata is never promoted to a portable field.

The encoder does not revalidate every trusted nested object. Construction and
port boundaries already guarantee their invariants.

## Trust Boundaries

Full structural validation occurs when data enters through:

- a portable wire decoder;
- public `Runtime` input;
- a model return;
- a tool/catalog/binding return;
- an approval or history policy return.

Inside the engine, frozen values are passed by reference and changed with
shallow structural copies. Serialization is not a defensive-copy mechanism.

## Contract Parity

For every portable top-level document, tests prove:

- schema-valid fixtures decode;
- schema-invalid and unknown-field fixtures fail;
- `decode(encode(value)) == value`;
- the implementation discriminator vocabulary exactly matches `contracts/v0`;
- conformance reports use the same codec rather than a second parser;
- codec modules do not perform network retrieval.

Adding or changing a portable field requires one atomic update to normative
documentation, schema, conformance fixtures, codec, implementation, and tests.
There is one reader and one writer for the active contract.

Decoding a trace establishes structural and domain validity only. A caller that treats
the trace as diagnostic evidence must call `verify_trace` after `decode_trace`; decode
success is not a verification result.

## Non-Goals

Kernel does not contain a reflection serializer, ORM, validation framework,
automatic schema generator, alternate-version reader, or provider codec.
Explicit functions are preferred when they make a portable aggregate and its
invariants visible.
