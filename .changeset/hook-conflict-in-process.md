---
'@workflow/core': patch
'@workflow/world': patch
---

Resolve a `hook.getConflict()` awaiter in the invocation that created the hook instead of re-invoking through the queue. The `hook_created` write now asks for the event-log delta since the replay's cursor (`CreateEventParams.sinceCursor`), and the runtime resumes the retained VM over the returned event — removing a delivery round-trip and a cold replay per awaited hook. Worlds that return no delta fall back to an incremental read, and `WORKFLOW_RETAINED_VM=0` restores the re-invocation.
