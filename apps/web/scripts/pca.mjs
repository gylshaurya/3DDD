import { chromium } from '@playwright/test';
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
for (const url of process.argv.slice(2)) {
  const page = await browser.newPage({ viewport: { width: 320, height: 200 } });
  await page.goto(`http://localhost:5199/viewer-test.html?hud=0&url=${url}`);
  await page.waitForFunction(() => window.__splat && (window.__splat.loaded || window.__splat.error), null, { timeout: 120000 });
  const r = await page.evaluate(() => {
    const v = window.__splat.viewer; const s = v.active.allSamples(); const c = v.bounds.center; const R = v.bounds.radius;
    let C = [0,0,0,0,0,0], n = 0;
    for (let i = 0; i < s.length; i += 4) { if (s[i+3] <= 0.25) continue; const x = s[i]-c[0], y = s[i+1]-c[1], z = s[i+2]-c[2]; if (Math.hypot(x,y,z) > R) continue;
      C[0]+=x*x; C[1]+=x*y; C[2]+=x*z; C[3]+=y*y; C[4]+=y*z; C[5]+=z*z; n++; }
    C = C.map(v => v/n);
    // Jacobi eigen
    let A = [[C[0],C[1],C[2]],[C[1],C[3],C[4]],[C[2],C[4],C[5]]], V = [[1,0,0],[0,1,0],[0,0,1]];
    for (let it = 0; it < 50; it++) { let p=0,q=1; for (const [a,b] of [[0,1],[0,2],[1,2]]) if (Math.abs(A[a][b]) > Math.abs(A[p][q])) {p=a;q=b;}
      if (Math.abs(A[p][q]) < 1e-12) break; const th = 0.5*Math.atan2(2*A[p][q], A[q][q]-A[p][p]); const cs=Math.cos(th), sn=Math.sin(th);
      for (let k=0;k<3;k++){ const akp=A[k][p], akq=A[k][q]; A[k][p]=cs*akp-sn*akq; A[k][q]=sn*akp+cs*akq; }
      for (let k=0;k<3;k++){ const apk=A[p][k], aqk=A[q][k]; A[p][k]=cs*apk-sn*aqk; A[q][k]=sn*apk+cs*aqk; }
      for (let k=0;k<3;k++){ const vkp=V[k][p], vkq=V[k][q]; V[k][p]=cs*vkp-sn*vkq; V[k][q]=sn*vkp+cs*vkq; } }
    const ev = [0,1,2].map(i => ({ val: A[i][i], vec: [V[0][i], V[1][i], V[2][i]] })).sort((a,b) => a.val-b.val);
    return ev.map(e => e.val.toFixed(3) + ' [' + e.vec.map(x => x.toFixed(2)).join(',') + ']').join(' | ');
  });
  console.log(url.padEnd(24), r);
  await page.close();
}
await browser.close();
