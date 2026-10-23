import {
  add, cross, dot, length, lookAtCV, normalize, perpendicular, scale, sub, type Vec3,
} from './math';

/** 'auto' estimates up from the scene (ground-plane PCA), falling back to y-down. */
export type UpAxis = 'auto' | 'y-down' | 'y-up' | 'z-up' | 'z-down' | Vec3;

export function upVector(up: UpAxis): Vec3 {
  if (Array.isArray(up)) return normalize(up);
  switch (up) {
    case 'y-up': return [0, 1, 0];
    case 'z-up': return [0, 0, 1];
    case 'z-down': return [0, 0, -1];
    case 'auto':
    case 'y-down':
    default: return [0, -1, 0];
  }
}

interface OrbitState {
  target: Vec3;
  yaw: number;
  pitch: number;
  distance: number;
}

const PITCH_LIMIT = Math.PI / 2 - 0.02;

/**
 * Damped orbit camera with an optional fly mode. "Goal" state follows input immediately;
 * the rendered state eases toward it each frame.
 */
export class OrbitCamera {
  fovY = (50 * Math.PI) / 180;
  up: Vec3;
  /** basis spanning the plane perpendicular to `up` */
  private e1: Vec3;
  private e2: Vec3;
  private goal: OrbitState = { target: [0, 0, 0], yaw: 0, pitch: 0.2, distance: 5 };
  private cur: OrbitState = { target: [0, 0, 0], yaw: 0, pitch: 0.2, distance: 5 };
  autoOrbit = false;
  autoOrbitSpeed = 0.25; // rad/s
  flyMode = false;
  /** Smallest/largest allowed orbit distance; set from scene bounds. */
  minDistance = 0.01;
  maxDistance = 1e4;
  sceneRadius = 1;

  constructor(up: UpAxis = 'y-down') {
    this.up = upVector(up);
    this.e1 = perpendicular(this.up);
    this.e2 = cross(this.up, this.e1);
  }

  setUp(up: UpAxis): void {
    this.up = upVector(up);
    this.e1 = perpendicular(this.up);
    this.e2 = cross(this.up, this.e1);
  }

  private eyeOf(s: OrbitState): Vec3 {
    const cp = Math.cos(s.pitch);
    const dir: Vec3 = add(
      add(scale(this.e1, cp * Math.cos(s.yaw)), scale(this.e2, cp * Math.sin(s.yaw))),
      scale(this.up, Math.sin(s.pitch)),
    );
    return add(s.target, scale(dir, s.distance));
  }

  get eye(): Vec3 {
    return this.eyeOf(this.cur);
  }

  get target(): Vec3 {
    return this.cur.target;
  }

  viewMatrix(): Float32Array {
    return lookAtCV(this.eye, this.cur.target, this.up);
  }

  /** Frame a sphere; `snap` skips easing. */
  fit(center: Vec3, radius: number, snap = true, sceneRadius = radius): void {
    const dist = (radius / Math.sin(this.fovY / 2)) * 0.9;
    this.sceneRadius = sceneRadius;
    this.minDistance = radius * 0.02;
    this.maxDistance = sceneRadius * 50;
    this.goal = { target: [...center] as Vec3, yaw: this.goal.yaw, pitch: 0.25, distance: dist };
    if (snap) this.cur = { ...this.goal, target: [...center] as Vec3 };
  }

  /** Set the view from an explicit eye + target (e.g. a training camera). */
  lookFrom(eye: Vec3, target: Vec3, snap = true): void {
    const d = sub(eye, target);
    const distance = length(d) || 1;
    const n = scale(d, 1 / distance);
    const pitch = Math.asin(Math.max(-1, Math.min(1, dot(n, this.up))));
    const yaw = Math.atan2(dot(n, this.e2), dot(n, this.e1));
    this.goal = { target: [...target] as Vec3, yaw, pitch, distance };
    if (snap) this.cur = { ...this.goal, target: [...target] as Vec3 };
  }

  orbit(dYaw: number, dPitch: number): void {
    if (this.flyMode) {
      // rotate the view around the eye: move the target instead of the eye
      const eye = this.eyeOf(this.goal);
      this.goal.yaw -= dYaw;
      this.goal.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.goal.pitch - dPitch));
      const newEye = this.eyeOf(this.goal);
      this.goal.target = add(this.goal.target, sub(eye, newEye));
      return;
    }
    this.goal.yaw += dYaw;
    this.goal.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.goal.pitch + dPitch));
  }

  /** Pan in screen space; dx, dy in fractions of the viewport height. */
  pan(dx: number, dy: number): void {
    const eye = this.eyeOf(this.goal);
    const f = normalize(sub(this.goal.target, eye));
    const r = normalize(cross(f, this.up));
    const d = cross(f, r); // screen-down in world
    const s = 2 * this.goal.distance * Math.tan(this.fovY / 2);
    this.goal.target = add(this.goal.target, add(scale(r, -dx * s), scale(d, -dy * s)));
  }

  /** Multiplicative zoom; factor < 1 moves closer. */
  zoom(factor: number): void {
    this.goal.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.goal.distance * factor));
  }

  /** Fly-mode translation in camera space (right, up, forward), in scene-radius units. */
  move(right: number, upAmt: number, forward: number): void {
    const eye = this.eyeOf(this.goal);
    const f = normalize(sub(this.goal.target, eye));
    const r = normalize(cross(f, this.up));
    const step = this.sceneRadius;
    const delta = add(add(scale(r, right * step), scale(this.up, upAmt * step)), scale(f, forward * step));
    this.goal.target = add(this.goal.target, delta);
  }

  /** Advance easing; returns true while still moving. */
  update(dt: number): boolean {
    if (this.autoOrbit) this.goal.yaw += this.autoOrbitSpeed * dt;
    const k = 1 - Math.exp(-dt * 12);
    const c = this.cur, g = this.goal;
    const before = c.yaw + c.pitch + c.distance + c.target[0] + c.target[1] + c.target[2];
    c.yaw += (g.yaw - c.yaw) * k;
    c.pitch += (g.pitch - c.pitch) * k;
    c.distance += (g.distance - c.distance) * k;
    c.target = [
      c.target[0] + (g.target[0] - c.target[0]) * k,
      c.target[1] + (g.target[1] - c.target[1]) * k,
      c.target[2] + (g.target[2] - c.target[2]) * k,
    ];
    const after = c.yaw + c.pitch + c.distance + c.target[0] + c.target[1] + c.target[2];
    const moving =
      Math.abs(g.yaw - c.yaw) > 1e-5 ||
      Math.abs(g.pitch - c.pitch) > 1e-5 ||
      Math.abs(g.distance - c.distance) > 1e-5 * g.distance ||
      length(sub(g.target, c.target)) > 1e-5 * g.distance;
    return moving || after !== before || this.autoOrbit;
  }
}

/** Mouse / touch / trackpad / keyboard input → OrbitCamera. */
export class CameraControls {
  private pointers = new Map<number, { x: number; y: number }>();
  private lastPinch = 0;
  private lastMid: { x: number; y: number } | null = null;
  private keys = new Set<string>();
  private readonly cleanup: (() => void)[] = [];
  onInteract: (() => void) | null = null;
  onKeyAction: ((action: 'fit' | 'autoOrbit' | 'fly') => void) | null = null;

  private readonly el: HTMLElement;
  private readonly cam: OrbitCamera;

  constructor(el: HTMLElement, cam: OrbitCamera) {
    this.el = el;
    this.cam = cam;
    el.style.touchAction = 'none';
    const on = <K extends keyof HTMLElementEventMap>(t: K, f: (e: HTMLElementEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      el.addEventListener(t, f as EventListener, opts);
      this.cleanup.push(() => el.removeEventListener(t, f as EventListener));
    };
    const onWin = <K extends keyof WindowEventMap>(t: K, f: (e: WindowEventMap[K]) => void) => {
      window.addEventListener(t, f as EventListener);
      this.cleanup.push(() => window.removeEventListener(t, f as EventListener));
    };
    if (!el.hasAttribute('tabindex')) el.tabIndex = 0;

    on('contextmenu', (e) => e.preventDefault());
    on('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.lastPinch = 0;
      this.lastMid = null;
      this.interact();
    });
    on('pointermove', (e) => this.onMove(e));
    const up = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      this.lastPinch = 0;
      this.lastMid = null;
    };
    on('pointerup', up);
    on('pointercancel', up);
    on('wheel', (e) => {
      e.preventDefault();
      this.interact();
      // ctrlKey wheel = trackpad pinch on macOS
      const amt = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      this.cam.zoom(Math.exp(amt * (e.ctrlKey ? 0.01 : 0.0015)));
    }, { passive: false });
    on('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === 'f') { this.onKeyAction?.('fly'); return; }
      if (k === 'r') { this.onKeyAction?.('fit'); return; }
      if (k === 'o') { this.onKeyAction?.('autoOrbit'); return; }
      if ('wasdqe'.includes(k) || k.startsWith('arrow')) {
        this.keys.add(k);
        this.interact();
        e.preventDefault();
      }
    });
    on('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    onWin('blur', () => this.keys.clear());
  }

  private interact(): void {
    this.onInteract?.();
  }

  private onMove(e: PointerEvent): void {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const h = this.el.clientHeight || 1;
    const dx = (e.clientX - p.x) / h;
    const dy = (e.clientY - p.y) / h;
    p.x = e.clientX;
    p.y = e.clientY;

    if (this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      if (this.lastPinch > 0) this.cam.zoom(this.lastPinch / Math.max(dist, 1));
      if (this.lastMid) this.cam.pan((mid.x - this.lastMid.x) / h, (mid.y - this.lastMid.y) / h);
      this.lastPinch = dist;
      this.lastMid = mid;
      return;
    }
    const panning = e.buttons === 2 || e.buttons === 4 || e.shiftKey || e.metaKey;
    if (panning) this.cam.pan(dx, dy);
    else this.cam.orbit(-dx * Math.PI * 1.2, dy * Math.PI * 1.2);
    this.interact();
  }

  /** Apply held keys; returns true if anything moved. */
  tick(dt: number): boolean {
    if (this.keys.size === 0) return false;
    const s = dt * 0.8;
    let r = 0, u = 0, f = 0;
    if (this.keys.has('w') || this.keys.has('arrowup')) f += s;
    if (this.keys.has('s') || this.keys.has('arrowdown')) f -= s;
    if (this.keys.has('d') || this.keys.has('arrowright')) r += s;
    if (this.keys.has('a') || this.keys.has('arrowleft')) r -= s;
    if (this.keys.has('e')) u += s;
    if (this.keys.has('q')) u -= s;
    this.cam.move(r, u, f);
    return true;
  }

  dispose(): void {
    for (const c of this.cleanup) c();
  }
}
