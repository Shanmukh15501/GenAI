# Human-in-the-Loop — LangGraph Flow

How a chatbot pauses for human review, persists state to MongoDB, and resumes when an admin provides an answer.

User session

Graph node

Interrupt / pause

MongoDB checkpoint

Admin session

Graph nodes

- **chatbot** — calls the LLM with bound tools on each turn
- **tools** — runs `human_assistance()`, which triggers `interrupt()`
- **tools_condition** — routes to tools if the LLM made a tool call, else ends

Interrupt mechanism

- `interrupt()` serialises the entire graph state to MongoDB and halts execution
- The same `thread_id` identifies the paused run across sessions
- Nothing is lost — messages, tool calls, and tool results are all checkpointed

MongoDB checkpointer

- Acts as durable storage between the user and admin sessions
- `get_state(config)` retrieves the snapshot and pending interrupts
- Enables the admin to run in a completely separate process

Admin resume

- `Command(resume={"data": …})` injects the human answer
- The graph picks up exactly where it paused — inside `human_assistance()`
- Execution continues: the LLM receives the answer and replies to the user