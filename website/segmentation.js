/* Native-grid analyzed surfaces. Reduced mesh is for visualization only. */
(() => {
  'use strict';
  const section = document.querySelector('.segmentation');
  if (!section) return;
  const stages = [...section.querySelectorAll('[data-stage]')];
  const motion = section.querySelector('#segmentation-motion');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const canvas = section.querySelector('#analyzed-model');
  const view = section.querySelector('.model-view');
  const status = section.querySelector('#model-status');
  let index = 0, paused = reduced.matches, visible = false, timer = null;
  let gl, program, uniforms, count = 0, selected = 1;
  let frame = null, last = 0, stageValue = 0, yaw = -.35, pitch = .52, zoom = 1;
  let dragged = false, dragging = false, pointerX = 0, pointerY = 0;
  const vertexSource = `#version 300 es
    in vec3 position; in vec3 normal; in float label;
    uniform vec2 angles; uniform float aspect; uniform float zoom;
    out vec3 vNormal; flat out float vLabel;
    vec3 rotate(vec3 p){
      float c=cos(angles.x),s=sin(angles.x);
      p=vec3(c*p.x+s*p.z,p.y,-s*p.x+c*p.z);
      c=cos(angles.y);s=sin(angles.y);
      return vec3(p.x,c*p.y-s*p.z,s*p.y+c*p.z);
    }
    void main(){
      vec3 p=rotate(position);float z=p.z-3.4;
      float fit=min(aspect,1.0);
      gl_Position=vec4(p.x*2.5*zoom*fit/aspect,p.y*2.5*zoom*fit,-1.020202*z-.202020,-z);
      vNormal=rotate(normal);vLabel=label;
    }`;
  const fragmentSource = `#version 300 es
    precision highp float;
    in vec3 vNormal; flat in float vLabel;
    uniform float stage; uniform float selected;
    out vec4 color;
    void main(){
      bool vessel=vLabel<.5;
      vec3 base=vessel?vec3(.72,.36,.39):vec3(.39,.76,.49);
      vec3 palette=.52+.24*cos(vec3(0.0,2.1,4.2)+vLabel*2.399);
      vec3 labeled=vessel?base:palette;
      vec3 rgb=mix(base,labeled,clamp(stage,0.,1.));
      float isolate=clamp(stage-1.,0.,1.);
      if(!vessel&&abs(vLabel-selected)>.5){
        if(isolate>.995)discard;
        // Screen-door fade avoids opaque ghost surfaces while changing stages.
        float threshold=fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453);
        if(threshold<isolate)discard;
      }
      vec3 isolated=vessel?vec3(.23,.105,.125):vec3(.70,.84,.65);
      rgb=mix(rgb,isolated,isolate);
      vec3 n=normalize(vNormal);if(!gl_FrontFacing)n=-n;
      float diffuse=max(dot(n,normalize(vec3(-.3,.6,1.))),0.);
      float rim=pow(1.-abs(n.z),3.)*.12;
      float spec=pow(max(dot(reflect(-normalize(vec3(-.3,.6,1.)),n),vec3(0.,0.,1.)),0.),28.)*.1;
      color=vec4(rgb*(.30+.70*diffuse)+rim*rgb+spec,1.);
    }`;
  function shader(type, source) {
    const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function render() {
    if (!program || gl.isContextLost()) return;
    const bounds=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,1.5);
    const width=Math.max(1,Math.round(bounds.width*ratio)),height=Math.max(1,Math.round(bounds.height*ratio));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    gl.viewport(0,0,width,height);gl.clearColor(11/255,16/255,17/255,1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.uniform2f(uniforms.angles,yaw,pitch);gl.uniform1f(uniforms.aspect,width/height);
    gl.uniform1f(uniforms.zoom,zoom);gl.uniform1f(uniforms.stage,stageValue);gl.uniform1f(uniforms.selected,selected);
    gl.drawElements(gl.TRIANGLES,count,gl.UNSIGNED_INT,0);
  }
  function animate(now) {
    frame=null;
    if (!program || paused || !visible || document.hidden) return;
    const dt=last?Math.min((now-last)/1000,.06):0;last=now;
    if(!dragging&&!dragged)yaw+=dt*.045;
    stageValue+=(index-stageValue)*(1-Math.exp(-dt*4));
    render();frame=requestAnimationFrame(animate);
  }
  function schedule() {
    clearTimeout(timer);timer=null;
    if(frame!==null)cancelAnimationFrame(frame);frame=null;last=0;
    const playing=visible&&!paused&&!document.hidden&&!!program;
    section.classList.toggle('playing',playing);
    motion.textContent=paused?'Play walkthrough':'Pause walkthrough';
    motion.setAttribute('aria-pressed',String(paused));
    if(playing){timer=setTimeout(()=>select((index+1)%stages.length),6000);frame=requestAnimationFrame(animate);}
    else {stageValue=index;render();}
  }
  function select(next) {
    index=next;
    stages.forEach((button,i)=>{button.classList.toggle('selected',i===index);button.setAttribute('aria-pressed',String(i===index));});
    const descriptions=['Analyzed microglia and vessels in 3D','Segmentation components colored by label','Smaller labeled microglia component isolated with vessels in context'];
    canvas.setAttribute('aria-label',descriptions[index]+'. Drag or use arrow keys to rotate; use the zoom buttons to inspect.');
    schedule();
  }
  stages.forEach((button,i)=>button.addEventListener('click',()=>{paused=true;select(i);}));
  motion.addEventListener('click',()=>{paused=!paused;schedule();});
  reduced.addEventListener('change',()=>{paused=reduced.matches;schedule();});
  document.addEventListener('visibilitychange',schedule);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;schedule();},{threshold:.15}).observe(section);
  new ResizeObserver(render).observe(canvas);
  canvas.addEventListener('pointerdown',event=>{
    if(!program)return;
    dragging=true;dragged=true;pointerX=event.clientX;pointerY=event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove',event=>{
    if(!dragging)return;
    yaw+=(event.clientX-pointerX)*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+(event.clientY-pointerY)*.008));
    pointerX=event.clientX;pointerY=event.clientY;render();
  });
  ['pointerup','pointercancel','lostpointercapture'].forEach(name=>canvas.addEventListener(name,()=>{dragging=false;}));
  canvas.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
    event.preventDefault();dragged=true;
    if(event.key==='ArrowLeft')yaw-=.12;
    if(event.key==='ArrowRight')yaw+=.12;
    if(event.key==='ArrowUp')pitch=Math.max(-1.3,pitch-.12);
    if(event.key==='ArrowDown')pitch=Math.min(1.3,pitch+.12);
    render();
  });
  section.querySelector('#model-zoom-in').addEventListener('click',()=>{zoom=Math.min(2.5,zoom*1.15);render();});
  section.querySelector('#model-zoom-out').addEventListener('click',()=>{zoom=Math.max(.5,zoom/1.15);render();});
  section.querySelector('#model-reset').addEventListener('click',()=>{yaw=-.35;pitch=.52;zoom=1;dragged=false;render();});
  function fail() {
    view.classList.add('failed');view.setAttribute('aria-busy','false');
    status.textContent='3D render · interactive view unavailable';
    paused=true;schedule();motion.disabled=true;
  }
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();fail();});
  async function load() {
    try {
      gl=canvas.getContext('webgl2',{alpha:false,antialias:true});
      if(!gl)throw new Error('WebGL2 unavailable');
      const [meshResponse,metaResponse]=await Promise.all([fetch('assets/analyzed-cortex.bin'),fetch('assets/analyzed-cortex.json?v=10')]);
      if(!meshResponse.ok||!metaResponse.ok)throw new Error('Model download failed');
      const [buffer,meta]=await Promise.all([meshResponse.arrayBuffer(),metaResponse.json()]);
      const header=new DataView(buffer),vertices=header.getUint32(0,true);count=header.getUint32(4,true);
      if(buffer.byteLength!==8+vertices*28+count*4)throw new Error('Invalid mesh');
      selected=meta.selected_cell;
      program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertexSource));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragmentSource));gl.linkProgram(program);
      if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
      gl.useProgram(program);gl.bindVertexArray(gl.createVertexArray());
      const vertexBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vertexBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(buffer,8,vertices*7),gl.STATIC_DRAW);
      [['position',3,0],['normal',3,12],['label',1,24]].forEach(([name,size,offset])=>{const a=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,size,gl.FLOAT,false,28,offset);});
      const indexBuffer=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(buffer,8+vertices*28,count),gl.STATIC_DRAW);
      uniforms=Object.fromEntries(['angles','aspect','zoom','stage','selected'].map(name=>[name,gl.getUniformLocation(program,name)]));
      gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
      section.querySelector('#model-count').textContent=`${meta.cell_count} labeled components`;
      view.classList.add('ready');view.setAttribute('aria-busy','false');status.textContent='Drag to rotate';
      render();schedule();
    } catch(error) { console.error('Analyzed 3D model:',error);program=null;fail(); }
  }
  select(0);
  // Download only when the model approaches the viewport.
  const loader=new IntersectionObserver(entries=>{if(entries[0].isIntersecting){loader.disconnect();load();}},{rootMargin:'300px'});
  loader.observe(view);
})();
