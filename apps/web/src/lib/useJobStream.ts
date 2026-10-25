import { useEffect, useReducer } from 'react';
import { u, type Job, type JobEvent, type SceneMeta, type SfmStats, type Stage, type TrainProgress } from './api';

export interface LogLine {
  t: number;
  msg: string;
  error?: boolean;
}

export interface JobStreamState {
  job: Job | null;
  /** True once the server finished replaying past events and we are live. */
  live: boolean;
  logs: LogLine[];
  stageMsg: Partial<Record<Stage, string>>;
  stageProgress: Partial<Record<Stage, number>>;
  progress: TrainProgress | null;
  curve: { step: number; psnr: number; gaussians: number }[];
  thumb: string | null;
  frames: { count: number; urls: string[] } | null;
  sfm: { url: string; scene: SceneMeta; stats: SfmStats } | null;
  snapshot: { step: number; url: string; splats: number } | null;
  snapshotCount: number;
  compares: { url: string; held_out: boolean }[];
  final: { ply: string; splat: string } | null;
  connectionLost: boolean;
}

const initial: JobStreamState = {
  job: null, live: false, logs: [], stageMsg: {}, stageProgress: {}, progress: null, curve: [], thumb: null, frames: null,
  sfm: null, snapshot: null, snapshotCount: 0, compares: [], final: null, connectionLost: false,
};

/** Make file URLs in events absolute. */
function resolveUrls(ev: JobEvent): JobEvent {
  switch (ev.type) {
    case 'thumb':
    case 'sfm':
    case 'snapshot':
    case 'compare':
      return { ...ev, url: u(ev.url) };
    case 'frames':
      return { ...ev, urls: ev.urls.map(u) };
    case 'done':
      return { ...ev, ply: u(ev.ply), splat: u(ev.splat) };
    default:
      return ev;
  }
}

type Action = { kind: 'event'; ev: JobEvent | { type: 'replay_done' } } | { kind: 'reset' } | { kind: 'lost' };

function reduce(s: JobStreamState, a: Action): JobStreamState {
  if (a.kind === 'reset') return initial;
  if (a.kind === 'lost') return { ...s, connectionLost: true };
  const ev = a.ev;
  switch (ev.type) {
    case 'replay_done':
      return { ...s, live: true, connectionLost: false };
    case 'job':
      return { ...s, job: ev.job };
    case 'log':
      return { ...s, logs: [...s.logs.slice(-199), { t: ev.t, msg: ev.msg, error: ev.level === 'error' }] };
    case 'stage_progress':
      return {
        ...s,
        stageMsg: { ...s.stageMsg, [ev.stage]: ev.msg },
        stageProgress: { ...s.stageProgress, [ev.stage]: ev.progress },
      };
    case 'thumb':
      return { ...s, thumb: ev.url };
    case 'frames':
      return { ...s, frames: { count: ev.count, urls: ev.urls } };
    case 'sfm':
      return { ...s, sfm: { url: ev.url, scene: ev.scene, stats: ev.stats } };
    case 'train_progress': {
      const { t: _t, type: _type, ...p } = ev;
      const curve =
        p.psnr != null && p.gaussians != null
          ? [...s.curve, { step: p.step, psnr: p.psnr, gaussians: p.gaussians }]
          : s.curve;
      return {
        ...s,
        progress: p,
        curve,
        stageProgress: { ...s.stageProgress, train: p.step / p.steps },
      };
    }
    case 'snapshot':
      return { ...s, snapshot: { step: ev.step, url: ev.url, splats: ev.splats }, snapshotCount: s.snapshotCount + 1 };
    case 'compare':
      return { ...s, compares: [...s.compares, { url: ev.url, held_out: ev.held_out }] };
    case 'metrics':
      return s.job ? { ...s, job: { ...s.job, metrics: ev.metrics } } : s;
    case 'done':
      return { ...s, final: { ply: ev.ply, splat: ev.splat } };
    default:
      return s;
  }
}

/** Live job events over SSE. Reconnects and resumes after the last event received. */
export function useJobStream(jobId: string | null): JobStreamState {
  const [state, dispatch] = useReducer(reduce, initial);

  useEffect(() => {
    if (!jobId) return;
    dispatch({ kind: 'reset' });
    let es: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let lostToast: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    let finished = false;
    let received = 0; // events applied so far; reconnects resume from here (?since=N)

    const connect = () => {
      es = new EventSource(u(`/api/jobs/${jobId}/events?since=${received}`));
      es.onmessage = (m) => {
        const ev = JSON.parse(m.data as string) as JobEvent | { type: 'replay_done' };
        if (ev.type === 'replay_done') {
          clearTimeout(lostToast);
          dispatch({ kind: 'event', ev });
          return;
        }
        received++;
        if (ev.type === 'job' && ['done', 'failed', 'cancelled'].includes(ev.job.status)) finished = true;
        dispatch({ kind: 'event', ev: resolveUrls(ev) });
      };
      // The server closes each stream after 100 s, so reconnect and continue.
      es.onerror = () => {
        es?.close();
        if (closed || finished) return;
        // Only show the reconnect message if it takes more than 3 s.
        clearTimeout(lostToast);
        lostToast = setTimeout(() => dispatch({ kind: 'lost' }), 3000);
        retry = setTimeout(connect, 800);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      clearTimeout(lostToast);
      es?.close();
    };
  }, [jobId]);

  return state;
}

/** Seconds each stage gets in a replay. */
const REPLAY_STAGE_SECONDS: Record<Stage, number> = { frames: 4, sfm: 16, train: 45, export: 4 };

/** Replay time for each event. Each stage is scaled to its REPLAY_STAGE_SECONDS share. */
export function replaySchedule(events: JobEvent[], seconds: number): number[] {
  const budget = Object.values(REPLAY_STAGE_SECONDS).reduce((a, b) => a + b, 0);
  const k = seconds / budget;
  const starts: { stage: Stage; t: number }[] = [];
  for (const ev of events) if (ev.type === 'stage') starts.push({ stage: ev.stage, t: ev.t });
  const end = events[events.length - 1].t;
  let offset = 0;
  const segs = starts.map((st, i) => {
    const realEnd = i + 1 < starts.length ? starts[i + 1].t : end;
    const seg = { t0: st.t, t1: Math.max(realEnd, st.t + 1e-3), r0: offset, len: REPLAY_STAGE_SECONDS[st.stage] * k };
    offset += seg.len;
    return seg;
  });
  return events.map((ev) => {
    const seg = segs.findLast((x) => ev.t >= x.t0);
    if (!seg) return 0;
    return seg.r0 + (Math.min(ev.t, seg.t1) - seg.t0) / (seg.t1 - seg.t0) * seg.len;
  });
}

/** Replays a finished run from its saved event log. */
export function useJobReplay(jobId: string, seconds = 70, run = 0): JobStreamState {
  const [state, dispatch] = useReducer(reduce, initial);

  useEffect(() => {
    let cancelled = false;
    const timers: ReturnType<typeof setTimeout>[] = [];
    dispatch({ kind: 'reset' });
    fetch(u(`/files/jobs/${jobId}/events.jsonl`))
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error('No recording for this run'))))
      .then((text) => {
        const events = text
          .split('\n')
          .filter(Boolean)
          .map((l) => JSON.parse(l) as JobEvent);
        if (!events.length || cancelled) return;
        const at = replaySchedule(events, seconds);
        dispatch({ kind: 'event', ev: { type: 'replay_done' } });
        events.forEach((ev, i) =>
          timers.push(setTimeout(() => !cancelled && dispatch({ kind: 'event', ev: resolveUrls(ev) }), at[i] * 1000)),
        );
      })
      .catch(() => dispatch({ kind: 'lost' }));
    return () => {
      cancelled = true;
      timers.forEach(clearTimeout);
    };
  }, [jobId, seconds, run]);

  return state;
}
