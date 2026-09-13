# Gate 1b withdrawal record

Status: **withdrawn before execution**

Gate 1b was frozen for review and was never run. Source review found that its
affected-arm contract could not pass under the mechanism it intended to test:

- the synchronous NCCL reduce-scatter call may return on the CPU after adding a
  stream dependency, so rank 0 is expected to block later at `dist.barrier()`;
- the contract incorrectly required no reduce return from either rank;
- the affected wall-bound prediction also depended on rank 1 remaining alive
  during process-group teardown, but Gate 1b did not instrument that teardown;
- one decoded Flight Recorder dump could make a generic `missing_member` result
  ambiguous between "the other rank did not participate" and "the other rank
  did not dump."

`GATE1B_FREEZE.json` and all seven files it hashes remain unchanged as an audit
record. They must not be executed as the current protocol. Gate 1c replaces the
contract with separate mechanism, termination and capture gates.

