/* Procedural neurovascular illustration. Geometry is generated once. */
(() => {
  'use strict';
  const canvas = document.querySelector('#neural-field');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  document.documentElement.classList.add('js');
  const reveal = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('visible'); reveal.unobserve(entry.target); } });
  }, { threshold: .12 });
  document.querySelectorAll('.reveal').forEach(el => reveal.observe(el));
  const button = document.querySelector('#motion');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = reduced.matches, frame = null, t = 0, last = 0, onScreen = true;
  let width = 1, height = 1, ratio = 1, pointer = { x: 0, y: 0 }, aim = { x: 0, y: 0 };
  let seed = 97;
  const random = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
  const branches = [], cells = [];
  function branch(x, y, z, angle, length, depth, cell = false) {
    const points = [], bend = (random() - .5) * .8, dz = (random() - .5) * length * .85;
    for (let i = 0; i <= 20; i++) {
      const f = i / 20;
      points.push([x + Math.cos(angle + bend * f) * length * f, y + Math.sin(angle + bend * f) * length * f, z + dz * f]);
    }
    branches.push({ points, depth, cell, phase: random() });
    if (depth > 0) {
      const end = points.at(-1);
      [-.52, .5].forEach(a => branch(...end, angle + bend + a + (random() - .5) * .3, length * (.64 + random() * .14), depth - 1, cell));
    }
  }
  branch(320, 580, 0, -1.78, 154, 6);

  [[160,185,-35],[385,268,45],[220,410,60]].forEach(([x,y,z]) => {
    cells.push([x,y,z]);
    for (let i = 0; i < 7; i++) branch(x,y,z,i*Math.PI*2/7,20+random()*22,3,true);
  });
  const dust = Array.from({length: 0}, () => [random()*600, random()*620, (random()-.5)*180, random()]);
  // Fit every branch, including fine tips, into the illustration's safe area.
  const all = branches.flatMap(b => b.points);
  const minX = Math.min(...all.map(p => p[0])), maxX = Math.max(...all.map(p => p[0]));
  const minY = Math.min(...all.map(p => p[1])), maxY = Math.max(...all.map(p => p[1]));
  const centerX = (minX+maxX)/2, centerY = (minY+maxY)/2;
  function draw() {
    if (width < 40 || height < 70) return;
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.clearRect(0,0,width,height);
    const scale = Math.min((width-40)/(maxX-minX), (height-70)/(maxY-minY)) * .88;
    const yaw = Math.sin(t*.12)*.11 + pointer.x*.075;
    const pitch = Math.cos(t*.09)*.055 + pointer.y*.045;
    const roll = Math.sin(t*.1)*.015;
    const intro = reduced.matches ? 1 : Math.min(1, .82 + t*.09);
    function project([x,y,z]) {
      x -= centerX; y -= centerY;
      const rx = x*Math.cos(yaw) + z*Math.sin(yaw), rz = z*Math.cos(yaw) - x*Math.sin(yaw);
      const ry = y*Math.cos(pitch) - rz*Math.sin(pitch), depth = y*Math.sin(pitch) + rz*Math.cos(pitch);
      const perspective = 850/(850-depth);
      return [width/2+(rx*Math.cos(roll)-ry*Math.sin(roll))*scale*perspective*intro,
              (height-24)/2+(rx*Math.sin(roll)+ry*Math.cos(roll))*scale*perspective*intro+Math.sin(t*.21)*4,
              depth, perspective];
    }
    dust.forEach(([x,y,z,phase]) => {
      const p = project([x,y+Math.sin(t*.18+phase*6)*8,z]);
      ctx.fillStyle = `rgba(183,201,182,${.05+.05*Math.sin(t*.4+phase*6)**2})`;
      ctx.beginPath();ctx.arc(p[0],p[1],.7*p[3],0,Math.PI*2);ctx.fill();
    });
    const ordered = branches.map(b => ({b, points:b.points.map(project)}));
    ordered.sort((a,b) => a.points[0][2]-b.points[0][2]);
    ctx.lineCap='round'; ctx.lineJoin='round';
    ordered.forEach(({b,points}) => {
      const depth = points[0][2], fade = Math.max(.35,Math.min(1,.72+depth/280));
      const opacity = (.21+b.depth*.095)*fade;
      ctx.strokeStyle = b.cell ? `rgba(158,190,164,${opacity})` : `rgba(189,139,145,${opacity})`;
      ctx.lineWidth = (b.cell ? .65+b.depth*.45 : .65+Math.pow(b.depth,1.5)*.7)*scale*points[0][3];
      ctx.beginPath();points.forEach(([x,y],i) => i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();
      if (!b.cell && b.depth>2) {
        ctx.strokeStyle = `rgba(223,184,183,${opacity*.3})`;ctx.lineWidth*=.28;ctx.stroke();
      }
      if (b.depth>1) {
        const f=(t*.055+b.phase)%1, index=f*20, low=Math.floor(index), a=points[low], z=points[Math.min(low+1,20)], mix=index-low;
        ctx.fillStyle=b.cell?`rgba(208,225,197,${fade*.8})`:`rgba(234,193,189,${fade*.8})`;
        ctx.beginPath();ctx.arc(a[0]+(z[0]-a[0])*mix,a[1]+(z[1]-a[1])*mix,(b.cell?1:1.5)*scale,0,Math.PI*2);ctx.fill();
      }
    });
    cells.forEach(cell => {
      const [x,y,z,perspective]=project(cell), radius=4*scale*perspective;
      ctx.fillStyle=`rgba(169,197,164,${.65+z/500})`;
      ctx.beginPath();ctx.ellipse(x,y,radius,radius*1.4,.4+Math.sin(t*.12)*.1,0,Math.PI*2);ctx.fill();
    });
  }
  function animate(now) {
    frame=null;
    if(paused||document.hidden||!onScreen)return;
    const dt=last?Math.min((now-last)/1000,.05):0; t+=dt;last=now;
    const smooth=1-Math.exp(-dt*3);pointer.x+=(aim.x-pointer.x)*smooth;pointer.y+=(aim.y-pointer.y)*smooth;
    draw();frame=requestAnimationFrame(animate);
  }
  function sync() {
    if(frame!==null)cancelAnimationFrame(frame);frame=null;last=0;
    const label=paused?'Resume animation':'Pause animation';
    button.setAttribute('aria-label',label);button.title=label;button.setAttribute('aria-pressed',String(paused));
    button.firstElementChild.textContent=paused?'▷':'Ⅱ';draw();
    if(!paused&&!document.hidden&&onScreen)frame=requestAnimationFrame(animate);
  }
  button.addEventListener('click',()=>{paused=!paused;sync();});
  reduced.addEventListener('change',()=>{paused=reduced.matches;sync();});
  canvas.addEventListener('pointermove',event=>{if(event.pointerType==='touch')return;const r=canvas.getBoundingClientRect();aim={x:(event.clientX-r.left)/r.width-.5,y:(event.clientY-r.top)/r.height-.5};});
  canvas.addEventListener('pointerleave',()=>{aim={x:0,y:0};});
  document.addEventListener('visibilitychange',sync);
  new ResizeObserver(entries=>{const r=entries[0].contentRect;width=r.width;height=r.height;ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);draw();}).observe(canvas);
  new IntersectionObserver(entries=>{onScreen=entries[0].isIntersecting;sync();}).observe(canvas);
  sync();
})();
