import { describe, expect, it } from 'vitest';
import { OrbitCamera } from './camera';
import { lookAtCV, transformPoint, type Vec3 } from './math';

describe('lookAtCV', () => {
  it('maps the eye to the origin and the target onto +z (COLMAP camera convention)', () => {
    const eye: Vec3 = [1, -2, 3];
    const target: Vec3 = [4, 0, -1];
    const m = lookAtCV(eye, target, [0, -1, 0]);
    const e = transformPoint(m, eye);
    const t = transformPoint(m, target);
    e.forEach((v) => expect(v).toBeCloseTo(0, 5));
    expect(t[0]).toBeCloseTo(0, 5);
    expect(t[1]).toBeCloseTo(0, 5);
    expect(t[2]).toBeCloseTo(Math.hypot(3, 2, -4), 5);
  });
  it('puts world-up above the image center (negative camera y)', () => {
    const m = lookAtCV([0, 0, -5], [0, 0, 0], [0, -1, 0]);
    const p = transformPoint(m, [0, -1, 0]); // a point "up" in a y-down world
    expect(p[1]).toBeLessThan(0);
  });
});

describe('OrbitCamera', () => {
  it('lookFrom reproduces the requested eye', () => {
    const cam = new OrbitCamera('y-down');
    cam.lookFrom([2, -1, 3], [0.5, 0, 0], true);
    const e = cam.eye;
    expect(e[0]).toBeCloseTo(2, 5);
    expect(e[1]).toBeCloseTo(-1, 5);
    expect(e[2]).toBeCloseTo(3, 5);
  });
  it('eases toward the goal and settles', () => {
    const cam = new OrbitCamera('y-down');
    cam.fit([0, 0, 0], 1, true);
    cam.zoom(0.5);
    let moving = true;
    for (let i = 0; i < 600 && moving; i++) moving = cam.update(1 / 60);
    expect(moving).toBe(false);
  });
  it('clamps pitch so the camera never flips over the pole', () => {
    const cam = new OrbitCamera('z-up');
    cam.fit([0, 0, 0], 1, true);
    cam.orbit(0, 100);
    for (let i = 0; i < 300; i++) cam.update(1 / 60);
    const e = cam.eye;
    expect(Math.hypot(e[0], e[1])).toBeGreaterThan(1e-3);
  });
});
