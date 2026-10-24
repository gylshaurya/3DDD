import { describe, expect, it } from 'vitest';
import type { JobEvent } from './api';
import { replaySchedule } from './useJobStream';

const ev = (t: number, extra: Record<string, unknown>) => ({ t, ...extra }) as unknown as JobEvent;

describe('replaySchedule', () => {
  it('gives each stage its own time budget regardless of real duration', () => {
    const events = [
      ev(0, { type: 'stage', stage: 'frames' }),
      ev(2, { type: 'stage', stage: 'sfm' }),
      ev(100, { type: 'log', msg: 'mid sfm' }),
      ev(200, { type: 'stage', stage: 'train' }),
      ev(500, { type: 'stage', stage: 'export' }),
      ev(501, { type: 'done' }),
    ];
    const at = replaySchedule(events, 69);
    expect(at[0]).toBe(0);
    expect(at[1]).toBeCloseTo(4); // frames squeezed to 4 s
    expect(at[2]).toBeCloseTo(4 + 8, 0); // halfway through sfm -> halfway through its 16 s
    expect(at[3]).toBeCloseTo(20);
    expect(at[4]).toBeCloseTo(65);
    expect(at[5]).toBeCloseTo(69);
    for (let i = 1; i < at.length; i++) expect(at[i]).toBeGreaterThanOrEqual(at[i - 1]);
  });
});
