---
layout: page
title: 2PC with Multi-Paxos
description: CSE 535 Distributed System course project. A distributed bank application using 2 phase commit over multi-paxos implemented in C++.
img: assets/img/12.jpg
importance: 1
category: work
related_publications: false
---

Code at [F25-CSE535/2pc-cyz14](https://github.com/F25-CSE535/2pc-cyz14).

## Review Notes for distributed systems

Transactions: Concurrent Control, Recovery

Conflict serializable: dependency graph acyclic

Isolation level

- Dirty reads
- non-repeatable reads
- phantom reads

Timestamp ordering: use R-TS/W-TS

- Writes: TS(T_i) >= R-TS(x) and W-TS(x)
- Reads: TS(T_i) >= W-TS(x)

ACID:

- Atomicity (by Logging)
- Consistency
- Isolation (by Locking)
- Durability (by Logging)

2 Phase Locking: guarantee conflict serializability

- may result in cascading aborts & dirty reads
- Strong Strict 2PL: if a value written by a transaction is not read or overwritten by another tx until that tx finishes.

Cascading aborts

AHL: TEE 2f+1 nodes

SharPer: Flattened Sharding

Single Ledger: Resilient DB

Permissioned Blockchain paradigms:

- Order-Execute OX: tendermint, 2/3 votes
- Execute-Order-Validatin XOV: Hyperledger fabric, contenditious cause poor performance
- Order-parallel Execute OXII: Parblockchain, build dependency graph first
