# P1 launch operations, 2026-10-02

These standalone operational scripts record the P1 launch procedure. They do
not change the algorithm package. The frozen algorithm source is
`3ec2355fbc43993375daab06ec0ea39fb9d86991` (PR #19), with Linux source digest
`55e8137c4af517718394e43b96c477ac77b99cf6dc52b8844a9792ef90193ed7`.

The two campaign roots are `/workspace/zhouhanyu/campaign_p1_20261002_node0` and
`/workspace/zhouhanyu/campaign_p1_20261002_node1`. Each contains 240 formal P1
jobs, balanced across F6/M0, H/T, algorithm seeds and problem sizes. The nodes
share persistent storage; outputs remain separate. E15's stop request and the
E16 development checkout are preserved.

The sequence is:

1. Deploy the frozen source and `plan.json`; prepare both manifests with
   `scripts/run_p1_campaign.py prepare` from the algorithm source. The deployed
   manifests additionally clone each of the eight `THR8_*` JobSpecs into a
   corresponding `DB8_*` stage, keeping config, input and source hashes unchanged.
   The exact derived specs are included in the preflight archive. Restore only
   inputs/specs/manifest into fresh campaign roots when repeating acceptance;
   use a fresh control directory and do not import old completion flags.
2. Copy the operational helpers to `/workspace/zhouhanyu`. Run
   `p1_freeze_queue.py` once, before any formal run, to freeze paired blocks.
3. Start `p1_acceptance_node.py` separately on each container with `P1_NODE=0`
   or `P1_NODE=1`, then start `p1_acceptance_coordinator.py` once on node 0.
4. The coordinator runs isolated 1/2/4/8-worker throughput waves, simultaneous
   8-worker waves, 48 fixed 200-generation runs, three legacy 200-generation
   probes, and four 1200-second observer/RSS pilots. It writes manifest-bound
   launch gates only after these checks pass and the migration reserve fits.
5. P1 then drains in the background. Any new OOM or worker failure stops new
   dispatch and preserves evidence. Completed jobs are reused only after their
   checksums match; incomplete attempts are not automatically retried.
6. After the coordinator reports `P1_active`, `p1_collect_preflight.py` checks
   all completed preflight artifacts and packages their evidence plus frozen
   inputs. Copy the archive, manifest and checksums locally, verify every member,
   then use `p1_release.py` to publish and verify the preflight Release.
7. The one-time local `p1_finish_delivery.py` waits for both formal batches to
   drain. It invokes `p1_collect_finished.py`, validates planned/completed keys,
   copies the 500-MiB parts locally, verifies every part and tar member, then
   publishes and verifies the raw-result Release. It does not retry algorithms
   or remove server originals. Failure stops delivery and leaves a local state
   record. An unreachable SSH endpoint is removed from the watcher, not retried.
   Scientific analysis and the later Notion result synthesis remain pending.

The readable Git copies have formatting/import-order cleanup. The exact deployed
bytes and their SHA256 values remain in the campaign manifests and preflight
archive. `helper_provenance.json` records the original and readable-copy hashes,
and validates that formatting did not change executable statements or imports.
The provenance includes normalized LF hashes for Linux/Git verification as well
as original/local byte hashes, since Windows checkouts may use CRLF.
Do not replace helpers or alter a manifest while its campaign is active.

The first four RSS pilots all completed with three checkpoints, 12 sampled
phenotypes per run and no missing samples. Offline diagnostic elapsed time was
29.7–38.9 seconds and maximum process RSS was 0.0798 GiB. These are engineering
acceptance observations, not completed P1 research findings. Online observer cost
is included in the time budget and is substantial for M0; do not combine these
timed data with older E15 logs or infer final algorithm superiority from P1.

Platform expiry remains unverified. This batch uses the existing conservative
2026-10-04 migration target and a 24-hour recovery reserve. Formal P1 results,
analysis, full-data Release delivery and the eventual Notion result update remain
pending until those runs complete.

The helpers contain deliberately frozen paths, commits and dates. A future dated
campaign must review and freeze those constants again; this operational record is
not an unrestricted generic launcher.

The final-result packer also passed an end-to-end synthetic integrity check:
497 members and four 1-MiB test parts verified, with two completed runs out of
480 correctly reported as partial. This checks packaging and failure accounting,
not algorithm performance. Production uses 500-MiB parts. The initial Windows
path-separator mismatch found by this check was corrected by recording POSIX tar
member names.
