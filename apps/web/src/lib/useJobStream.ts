import { useEffect, useReducer } from 'react';
import type { Job, JobEvent, SceneMeta, SfmStats, Stage, TrainProgress } from './api';

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

/** Subscribes to /api/jobs/:id/events (SSE). Replays history, then streams live; auto-reconnects. */
export function useJobStream(jobId: string | null): JobStreamState {
  const [state, dispatch] = useReducer(reduce, initial);

  useEffect(() => {
    if (!jobId) return;
    dispatch({ kind: 'reset' });
    let es: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let closed = false;
    let finished = false;

    const connect = () => {
      es = new EventSource(`/api/jobs/${jobId}/events`);
      es.onmessage = (m) => {
        const ev = JSON.parse(m.data as string) as JobEvent | { type: 'replay_done' };
        if (ev.type === 'job' && ['done', 'failed', 'cancelled'].includes(ev.job.status)) finished = true;
        dispatch({ kind: 'event', ev });
      };
      es.onerror = () => {
        es?.close();
        if (closed || finished) return;
        dispatch({ kind: 'lost' });
        // A reconnect replays the full history, so start from a clean slate to avoid duplicates.
        retry = setTimeout(() => {
          dispatch({ kind: 'reset' });
          connect();
        }, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      es?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  return state;
}

/**
 * Re-plays a finished run from its recorded event log, time-compressed to roughly `seconds`.
 * Dead air (long gaps such as the camera solve) is capped so the story keeps moving.
 */
export function useJobReplay(jobId: string, seconds = 60, run = 0): JobStreamState {
  const [state, dispatch] = useReducer(reduce, initial);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    dispatch({ kind: 'reset' });
    fetch(`/files/jobs/${jobId}/events.jsonl`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error('No recording for this run'))))
      .then((text) => {
        const events = text
          .split('\n')
          .filter(Boolean)
          .map((l) => JSON.parse(l) as JobEvent);
        if (!events.length || cancelled) return;
        const span = events[events.length - 1].t - events[0].t;
        const speed = Math.max(1, span / seconds);
        dispatch({ kind: 'event', ev: { type: 'replay_done' } });
        let i = 0;
        const step = () => {
          if (cancelled) return;
          dispatch({ kind: 'event', ev: events[i] });
          i++;
          if (i >= events.length) return;
          const gap = (events[i].t - events[i - 1].t) / speed;
          timer = setTimeout(step, Math.min(gap, 1.2) * 1000);
        };
        step();
      })
      .catch(() => dispatch({ kind: 'lost' }));
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId, seconds, run]);

  return state;
}
