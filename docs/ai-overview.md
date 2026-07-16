# AI / LLM context — VectorBridge

> Concise reference for humans and coding assistants.
> Do not invent APIs beyond this file and the package/repo source.
> Package: **`insight-vector-bridge` 0.1.0** · Import: **`insight_vector_bridge / per README`**

---

## 10-sentence project summary

1. Vector database migration using CHORUS Fabric tensor transport with encryption and post-migration semantic validation.
2. Primary users: Teams moving embeddings between vector databases with bandwidth and integrity constraints.
3. Core problem: HTTP/JSON vector migration wastes bandwidth and lacks semantic validation.
4. Install/use from the repository README — do not invent extra CLI flags here.
5. Key surface: pip install insight-vector-bridge — see README for source/target connectors.
6. Compared with: HTTP/REST JSON dumps · gzip REST · raw binary transfer.
7. When NOT to use: You are not migrating between vector DBs.
8. Read architecture.md for stack placement.
9. Prefer facts from README / existing docs over marketing inference.
10. If an API is not listed in README or source, assume it does not exist.

---

## Core concepts

See README for product-specific terms. Keep terminology consistent with that file.

---

## Key APIs

```
pip install insight-vector-bridge — see README for source/target connectors.
```

---

## Common use cases

- HTTP/JSON vector migration wastes bandwidth and lacks semantic validation.
- See README examples and any `examples/` folder in the repo.

---

## Migration guidance

Start from the closest tool in: HTTP/REST JSON dumps · gzip REST · raw binary transfer. Follow README install and examples. Do not invent migration scripts that are not in the repo.

---

## Limitations / when NOT to use

- You are not migrating between vector DBs.
- Do not invent capabilities beyond README and source.

---

## Frequently compared projects

| Notes |
|-------|
| HTTP/REST JSON dumps · gzip REST · raw binary transfer |

---

## Links

- [ai-overview.md](ai-overview.md) · [llm-context.md](llm-context.md) · [architecture.md](architecture.md)
- ../README.md
