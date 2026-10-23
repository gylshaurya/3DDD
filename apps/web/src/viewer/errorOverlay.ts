/** Replace `container`'s content with a readable "WebGPU unavailable" message. */
export function showWebGPUError(container: HTMLElement, err: unknown): void {
  const message = err instanceof Error ? err.message : String(err);
  const box = document.createElement('div');
  box.setAttribute('role', 'alert');
  box.style.cssText = [
    'position:absolute', 'inset:0', 'display:flex', 'flex-direction:column', 'align-items:center',
    'justify-content:center', 'gap:10px', 'padding:24px', 'text-align:center',
    'font:15px/1.5 system-ui,sans-serif', 'color:#e8e8ea', 'background:#0b0c0f',
  ].join(';');
  const title = document.createElement('strong');
  title.textContent = 'WebGPU is not available';
  title.style.fontSize = '18px';
  const detail = document.createElement('p');
  detail.textContent = message;
  detail.style.cssText = 'margin:0;max-width:440px;color:#a9abb3';
  const hint = document.createElement('p');
  hint.textContent = 'Try the latest Chrome or Edge on desktop, or Safari 26+ on macOS / iOS.';
  hint.style.cssText = 'margin:0;max-width:440px;color:#7c7f88;font-size:13px';
  box.append(title, detail);
  if (!/chrome|safari/i.test(message)) box.append(hint);
  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';
  container.appendChild(box);
}
