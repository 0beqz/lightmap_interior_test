/**
 * postprocessing v6.21.3 build Wed Apr 07 2021
 * https://github.com/vanruesc/postprocessing
 * Copyright 2021 Raoul van Rüschen
 * @license Zlib
 */
// src/core/ColorChannel.js
var ColorChannel = {
  RED: 0,
  GREEN: 1,
  BLUE: 2,
  ALPHA: 3
};

// src/core/Disposable.js
var Disposable = class {
  dispose() {
  }
};

// src/core/EffectComposer.js
import {
  DepthStencilFormat,
  DepthTexture,
  LinearFilter as LinearFilter4,
  RGBAFormat as RGBAFormat3,
  RGBFormat as RGBFormat4,
  UnsignedByteType as UnsignedByteType8,
  UnsignedIntType,
  UnsignedInt248Type,
  Vector2 as Vector213,
  WebGLMultisampleRenderTarget,
  WebGLRenderTarget as WebGLRenderTarget9
} from "../build/three.module.js";

// src/passes/AdaptiveLuminancePass.js
import {
  HalfFloatType,
  NearestFilter,
  RGBAFormat,
  WebGLRenderTarget as WebGLRenderTarget2
} from "../build/three.module.js";

// src/materials/AdaptiveLuminanceMaterial.js
import {NoBlending, ShaderMaterial, Uniform} from "../build/three.module.js";

// src/materials/glsl/adaptive-luminance/shader.frag
var shader_default = "uniform mediump sampler2D luminanceBuffer0;uniform lowp sampler2D luminanceBuffer1;uniform float minLuminance;uniform float deltaTime;uniform float tau;varying vec2 vUv;void main(){float l0=texture2D(luminanceBuffer0,vUv).r;\n#if __VERSION__ < 300\nfloat l1=texture2DLodEXT(luminanceBuffer1,vUv,MIP_LEVEL_1X1).r;\n#else\nfloat l1=textureLod(luminanceBuffer1,vUv,MIP_LEVEL_1X1).r;\n#endif\nl0=max(minLuminance,l0);l1=max(minLuminance,l1);float adaptedLum=l0+(l1-l0)*(1.0-exp(-deltaTime*tau));gl_FragColor.r=adaptedLum;}";

// src/materials/glsl/common/shader.vert
var shader_default2 = "varying vec2 vUv;void main(){vUv=position.xy*0.5+0.5;gl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/AdaptiveLuminanceMaterial.js
var AdaptiveLuminanceMaterial = class extends ShaderMaterial {
  constructor() {
    super({
      type: "AdaptiveLuminanceMaterial",
      defines: {
        MIP_LEVEL_1X1: "0.0"
      },
      uniforms: {
        luminanceBuffer0: new Uniform(null),
        luminanceBuffer1: new Uniform(null),
        minLuminance: new Uniform(0.01),
        deltaTime: new Uniform(0),
        tau: new Uniform(1)
      },
      fragmentShader: shader_default,
      vertexShader: shader_default2,
      blending: NoBlending,
      depthWrite: false,
      depthTest: false,
      extensions: {
        shaderTextureLOD: true
      }
    });
    this.toneMapped = false;
  }
};

// src/materials/BokehMaterial.js
import {NoBlending as NoBlending2, ShaderMaterial as ShaderMaterial2, Uniform as Uniform2, Vector2, Vector4} from "../build/three.module.js";

// src/materials/glsl/bokeh/shader.frag
var shader_default3 = "#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\nuniform lowp sampler2D cocBuffer;uniform vec2 texelSize;uniform float scale;\n#if PASS == 1\nuniform vec4 kernel64[32];\n#else\nuniform vec4 kernel16[8];\n#endif\nvarying vec2 vUv;void main(){\n#ifdef FOREGROUND\nvec2 CoCNearFar=texture2D(cocBuffer,vUv).rg;float CoC=CoCNearFar.r*scale;\n#else\nfloat CoC=texture2D(cocBuffer,vUv).g*scale;\n#endif\nif(CoC==0.0){gl_FragColor=texture2D(inputBuffer,vUv);}else{\n#ifdef FOREGROUND\nvec2 step=texelSize*max(CoC,CoCNearFar.g*scale);\n#else\nvec2 step=texelSize*CoC;\n#endif\n#if PASS == 1\nvec4 acc=vec4(0.0);for(int i=0;i<32;++i){vec4 kernel=kernel64[i];vec2 uv=step*kernel.xy+vUv;acc+=texture2D(inputBuffer,uv);uv=step*kernel.zw+vUv;acc+=texture2D(inputBuffer,uv);}gl_FragColor=acc/64.0;\n#else\nvec4 maxValue=texture2D(inputBuffer,vUv);for(int i=0;i<8;++i){vec4 kernel=kernel16[i];vec2 uv=step*kernel.xy+vUv;maxValue=max(texture2D(inputBuffer,uv),maxValue);uv=step*kernel.zw+vUv;maxValue=max(texture2D(inputBuffer,uv),maxValue);}gl_FragColor=maxValue;\n#endif\n}}";

// src/materials/BokehMaterial.js
var BokehMaterial = class extends ShaderMaterial2 {
  constructor(fill = false, foreground = false) {
    super({
      type: "BokehMaterial",
      defines: {
        PASS: fill ? "2" : "1"
      },
      uniforms: {
        kernel64: new Uniform2(null),
        kernel16: new Uniform2(null),
        inputBuffer: new Uniform2(null),
        cocBuffer: new Uniform2(null),
        texelSize: new Uniform2(new Vector2()),
        scale: new Uniform2(1)
      },
      fragmentShader: shader_default3,
      vertexShader: shader_default2,
      blending: NoBlending2,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    if (foreground) {
      this.defines.FOREGROUND = "1";
    }
    this.generateKernel();
  }
  generateKernel() {
    const GOLDEN_ANGLE = 2.39996323;
    const points64 = new Float32Array(128);
    const points16 = new Float32Array(32);
    let i64 = 0, i16 = 0;
    for (let i = 0; i < 80; ++i) {
      const theta = i * GOLDEN_ANGLE;
      const r = Math.sqrt(i) / Math.sqrt(80);
      const u = r * Math.cos(theta), v3 = r * Math.sin(theta);
      if (i % 5 === 0) {
        points16[i16++] = u;
        points16[i16++] = v3;
      } else {
        points64[i64++] = u;
        points64[i64++] = v3;
      }
    }
    const kernel64 = [];
    const kernel16 = [];
    for (let i = 0; i < 128; ) {
      kernel64.push(new Vector4(points64[i++], points64[i++], points64[i++], points64[i++]));
    }
    for (let i = 0; i < 32; ) {
      kernel16.push(new Vector4(points16[i++], points16[i++], points16[i++], points16[i++]));
    }
    this.uniforms.kernel64.value = kernel64;
    this.uniforms.kernel16.value = kernel16;
  }
  setTexelSize(x, y) {
    this.uniforms.texelSize.value.set(x, y);
  }
};

// src/materials/CircleOfConfusionMaterial.js
import {NoBlending as NoBlending3, PerspectiveCamera, ShaderMaterial as ShaderMaterial3, Uniform as Uniform3} from "../build/three.module.js";

// src/materials/glsl/circle-of-confusion/shader.frag
var shader_default4 = "#include <common>\n#include <packing>\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer;\n#else\nuniform mediump sampler2D depthBuffer;\n#endif\nuniform float focusDistance;uniform float focalLength;uniform float cameraNear;uniform float cameraFar;varying vec2 vUv;float readDepth(const in vec2 uv){\n#if DEPTH_PACKING == 3201\nreturn unpackRGBAToDepth(texture2D(depthBuffer,uv));\n#else\nreturn texture2D(depthBuffer,uv).r;\n#endif\n}void main(){float depth=readDepth(vUv);\n#ifdef PERSPECTIVE_CAMERA\nfloat viewZ=perspectiveDepthToViewZ(depth,cameraNear,cameraFar);float linearDepth=viewZToOrthographicDepth(viewZ,cameraNear,cameraFar);\n#else\nfloat linearDepth=depth;\n#endif\nfloat signedDistance=linearDepth-focusDistance;float magnitude=smoothstep(0.0,focalLength,abs(signedDistance));gl_FragColor.rg=vec2(step(signedDistance,0.0)*magnitude,step(0.0,signedDistance)*magnitude);}";

// src/materials/CircleOfConfusionMaterial.js
var CircleOfConfusionMaterial = class extends ShaderMaterial3 {
  constructor(camera) {
    super({
      type: "CircleOfConfusionMaterial",
      defines: {
        DEPTH_PACKING: "0"
      },
      uniforms: {
        depthBuffer: new Uniform3(null),
        focusDistance: new Uniform3(0),
        focalLength: new Uniform3(0),
        cameraNear: new Uniform3(0.3),
        cameraFar: new Uniform3(1e3)
      },
      fragmentShader: shader_default4,
      vertexShader: shader_default2,
      blending: NoBlending3,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.adoptCameraSettings(camera);
  }
  get depthPacking() {
    return Number(this.defines.DEPTH_PACKING);
  }
  set depthPacking(value) {
    this.defines.DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  adoptCameraSettings(camera = null) {
    if (camera !== null) {
      this.uniforms.cameraNear.value = camera.near;
      this.uniforms.cameraFar.value = camera.far;
      if (camera instanceof PerspectiveCamera) {
        this.defines.PERSPECTIVE_CAMERA = "1";
      } else {
        delete this.defines.PERSPECTIVE_CAMERA;
      }
      this.needsUpdate = true;
    }
  }
};

// src/materials/ColorEdgesMaterial.js
import {NoBlending as NoBlending4, ShaderMaterial as ShaderMaterial4, Uniform as Uniform4, Vector2 as Vector22} from "../build/three.module.js";

// src/materials/glsl/edge-detection/shader.frag
var shader_default5 = "varying vec2 vUv;varying vec2 vUv0;varying vec2 vUv1;\n#if EDGE_DETECTION_MODE != 0\nvarying vec2 vUv2;varying vec2 vUv3;varying vec2 vUv4;varying vec2 vUv5;\n#endif\n#if EDGE_DETECTION_MODE == 1\n#include <common>\n#endif\n#if EDGE_DETECTION_MODE == 0 || PREDICATION_MODE == 1\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer;\n#else\nuniform mediump sampler2D depthBuffer;\n#endif\nfloat readDepth(const in vec2 uv){\n#if DEPTH_PACKING == 3201\nreturn unpackRGBAToDepth(texture2D(depthBuffer,uv));\n#else\nreturn texture2D(depthBuffer,uv).r;\n#endif\n}vec3 gatherNeighbors(){float p=readDepth(vUv);float pLeft=readDepth(vUv0);float pTop=readDepth(vUv1);return vec3(p,pLeft,pTop);}\n#elif PREDICATION_MODE == 2\nuniform sampler2D predicationBuffer;vec3 gatherNeighbors(){float p=texture2D(predicationBuffer,vUv).r;float pLeft=texture2D(predicationBuffer,vUv0).r;float pTop=texture2D(predicationBuffer,vUv1).r;return vec3(p,pLeft,pTop);}\n#endif\n#if PREDICATION_MODE != 0\nvec2 calculatePredicatedThreshold(){vec3 neighbours=gatherNeighbors();vec2 delta=abs(neighbours.xx-neighbours.yz);vec2 edges=step(PREDICATION_THRESHOLD,delta);return PREDICATION_SCALE*EDGE_THRESHOLD*(1.0-PREDICATION_STRENGTH*edges);}\n#endif\n#if EDGE_DETECTION_MODE != 0\nuniform sampler2D inputBuffer;\n#endif\nvoid main(){\n#if EDGE_DETECTION_MODE == 0\nconst vec2 threshold=vec2(DEPTH_THRESHOLD);\n#elif PREDICATION_MODE != 0\nvec2 threshold=calculatePredicatedThreshold();\n#else\nconst vec2 threshold=vec2(EDGE_THRESHOLD);\n#endif\n#if EDGE_DETECTION_MODE == 0\nvec3 neighbors=gatherNeighbors();vec2 delta=abs(neighbors.xx-vec2(neighbors.y,neighbors.z));vec2 edges=step(threshold,delta);if(dot(edges,vec2(1.0))==0.0){discard;}gl_FragColor=vec4(edges,0.0,1.0);\n#elif EDGE_DETECTION_MODE == 1\nfloat l=linearToRelativeLuminance(texture2D(inputBuffer,vUv).rgb);float lLeft=linearToRelativeLuminance(texture2D(inputBuffer,vUv0).rgb);float lTop=linearToRelativeLuminance(texture2D(inputBuffer,vUv1).rgb);vec4 delta;delta.xy=abs(l-vec2(lLeft,lTop));vec2 edges=step(threshold,delta.xy);if(dot(edges,vec2(1.0))==0.0){discard;}float lRight=linearToRelativeLuminance(texture2D(inputBuffer,vUv2).rgb);float lBottom=linearToRelativeLuminance(texture2D(inputBuffer,vUv3).rgb);delta.zw=abs(l-vec2(lRight,lBottom));vec2 maxDelta=max(delta.xy,delta.zw);float lLeftLeft=linearToRelativeLuminance(texture2D(inputBuffer,vUv4).rgb);float lTopTop=linearToRelativeLuminance(texture2D(inputBuffer,vUv5).rgb);delta.zw=abs(vec2(lLeft,lTop)-vec2(lLeftLeft,lTopTop));maxDelta=max(maxDelta.xy,delta.zw);float finalDelta=max(maxDelta.x,maxDelta.y);edges.xy*=step(finalDelta,LOCAL_CONTRAST_ADAPTATION_FACTOR*delta.xy);gl_FragColor=vec4(edges,0.0,1.0);\n#elif EDGE_DETECTION_MODE == 2\nvec4 delta;vec3 c=texture2D(inputBuffer,vUv).rgb;vec3 cLeft=texture2D(inputBuffer,vUv0).rgb;vec3 t=abs(c-cLeft);delta.x=max(max(t.r,t.g),t.b);vec3 cTop=texture2D(inputBuffer,vUv1).rgb;t=abs(c-cTop);delta.y=max(max(t.r,t.g),t.b);vec2 edges=step(threshold,delta.xy);if(dot(edges,vec2(1.0))==0.0){discard;}vec3 cRight=texture2D(inputBuffer,vUv2).rgb;t=abs(c-cRight);delta.z=max(max(t.r,t.g),t.b);vec3 cBottom=texture2D(inputBuffer,vUv3).rgb;t=abs(c-cBottom);delta.w=max(max(t.r,t.g),t.b);vec2 maxDelta=max(delta.xy,delta.zw);vec3 cLeftLeft=texture2D(inputBuffer,vUv4).rgb;t=abs(c-cLeftLeft);delta.z=max(max(t.r,t.g),t.b);vec3 cTopTop=texture2D(inputBuffer,vUv5).rgb;t=abs(c-cTopTop);delta.w=max(max(t.r,t.g),t.b);maxDelta=max(maxDelta.xy,delta.zw);float finalDelta=max(maxDelta.x,maxDelta.y);edges*=step(finalDelta,LOCAL_CONTRAST_ADAPTATION_FACTOR*delta.xy);gl_FragColor=vec4(edges,0.0,1.0);\n#endif\n}";

// src/materials/glsl/edge-detection/shader.vert
var shader_default6 = "uniform vec2 texelSize;varying vec2 vUv;varying vec2 vUv0;varying vec2 vUv1;\n#if EDGE_DETECTION_MODE != 0\nvarying vec2 vUv2;varying vec2 vUv3;varying vec2 vUv4;varying vec2 vUv5;\n#endif\nvoid main(){vUv=position.xy*0.5+0.5;vUv0=vUv+texelSize*vec2(-1.0,0.0);vUv1=vUv+texelSize*vec2(0.0,-1.0);\n#if EDGE_DETECTION_MODE != 0\nvUv2=vUv+texelSize*vec2(1.0,0.0);vUv3=vUv+texelSize*vec2(0.0,1.0);vUv4=vUv+texelSize*vec2(-2.0,0.0);vUv5=vUv+texelSize*vec2(0.0,-2.0);\n#endif\ngl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/ColorEdgesMaterial.js
var ColorEdgesMaterial = class extends ShaderMaterial4 {
  constructor(texelSize = new Vector22()) {
    super({
      type: "ColorEdgesMaterial",
      defines: {
        EDGE_DETECTION_MODE: "2",
        LOCAL_CONTRAST_ADAPTATION_FACTOR: "2.0",
        EDGE_THRESHOLD: "0.1"
      },
      uniforms: {
        inputBuffer: new Uniform4(null),
        texelSize: new Uniform4(texelSize)
      },
      fragmentShader: shader_default5,
      vertexShader: shader_default6,
      blending: NoBlending4,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
  }
  setLocalContrastAdaptationFactor(factor) {
    this.defines.LOCAL_CONTRAST_ADAPTATION_FACTOR = factor.toFixed("2");
    this.needsUpdate = true;
  }
  setEdgeDetectionThreshold(threshold) {
    const t = Math.min(Math.max(threshold, 0.05), 0.5);
    this.defines.EDGE_THRESHOLD = t.toFixed("2");
    this.needsUpdate = true;
  }
};

// src/materials/ConvolutionMaterial.js
import {NoBlending as NoBlending5, ShaderMaterial as ShaderMaterial5, Uniform as Uniform5, Vector2 as Vector23} from "../build/three.module.js";

// src/materials/glsl/convolution/shader.frag
var shader_default7 = "#include <common>\n#include <dithering_pars_fragment>\n#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\nvarying vec2 vUv0;varying vec2 vUv1;varying vec2 vUv2;varying vec2 vUv3;void main(){vec4 sum=texture2D(inputBuffer,vUv0);sum+=texture2D(inputBuffer,vUv1);sum+=texture2D(inputBuffer,vUv2);sum+=texture2D(inputBuffer,vUv3);gl_FragColor=sum*0.25;\n#include <dithering_fragment>\n}";

// src/materials/glsl/convolution/shader.vert
var shader_default8 = "uniform vec2 texelSize;uniform vec2 halfTexelSize;uniform float kernel;uniform float scale;varying vec2 vUv0;varying vec2 vUv1;varying vec2 vUv2;varying vec2 vUv3;void main(){vec2 uv=position.xy*0.5+0.5;vec2 dUv=(texelSize*vec2(kernel)+halfTexelSize)*scale;vUv0=vec2(uv.x-dUv.x,uv.y+dUv.y);vUv1=vec2(uv.x+dUv.x,uv.y+dUv.y);vUv2=vec2(uv.x+dUv.x,uv.y-dUv.y);vUv3=vec2(uv.x-dUv.x,uv.y-dUv.y);gl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/ConvolutionMaterial.js
var ConvolutionMaterial = class extends ShaderMaterial5 {
  constructor(texelSize = new Vector23()) {
    super({
      type: "ConvolutionMaterial",
      uniforms: {
        inputBuffer: new Uniform5(null),
        texelSize: new Uniform5(new Vector23()),
        halfTexelSize: new Uniform5(new Vector23()),
        kernel: new Uniform5(0),
        scale: new Uniform5(1)
      },
      fragmentShader: shader_default7,
      vertexShader: shader_default8,
      blending: NoBlending5,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.setTexelSize(texelSize.x, texelSize.y);
    this.kernelSize = KernelSize.LARGE;
  }
  getKernel() {
    return kernelPresets[this.kernelSize];
  }
  setTexelSize(x, y) {
    this.uniforms.texelSize.value.set(x, y);
    this.uniforms.halfTexelSize.value.set(x, y).multiplyScalar(0.5);
  }
};
var kernelPresets = [
  new Float32Array([0, 0]),
  new Float32Array([0, 1, 1]),
  new Float32Array([0, 1, 1, 2]),
  new Float32Array([0, 1, 2, 2, 3]),
  new Float32Array([0, 1, 2, 3, 4, 4, 5]),
  new Float32Array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
];
var KernelSize = {
  VERY_SMALL: 0,
  SMALL: 1,
  MEDIUM: 2,
  LARGE: 3,
  VERY_LARGE: 4,
  HUGE: 5
};

// src/materials/CopyMaterial.js
import {NoBlending as NoBlending6, ShaderMaterial as ShaderMaterial6, Uniform as Uniform6} from "../build/three.module.js";

// src/materials/glsl/copy/shader.frag
var shader_default9 = "#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\nuniform float opacity;varying vec2 vUv;void main(){vec4 texel=texture2D(inputBuffer,vUv);gl_FragColor=opacity*texel;\n#include <encodings_fragment>\n}";

// src/materials/CopyMaterial.js
var CopyMaterial = class extends ShaderMaterial6 {
  constructor() {
    super({
      type: "CopyMaterial",
      uniforms: {
        inputBuffer: new Uniform6(null),
        opacity: new Uniform6(1)
      },
      fragmentShader: shader_default9,
      vertexShader: shader_default2,
      blending: NoBlending6,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
  }
};

// src/materials/DepthComparisonMaterial.js
import {NoBlending as NoBlending7, PerspectiveCamera as PerspectiveCamera2, ShaderMaterial as ShaderMaterial7, Uniform as Uniform7} from "../build/three.module.js";

// src/materials/glsl/depth-comparison/shader.frag
var shader_default10 = "#include <packing>\n#include <clipping_planes_pars_fragment>\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer;\n#else\nuniform mediump sampler2D depthBuffer;\n#endif\nuniform float cameraNear;uniform float cameraFar;varying float vViewZ;varying vec4 vProjTexCoord;void main(){\n#include <clipping_planes_fragment>\nvec2 projTexCoord=(vProjTexCoord.xy/vProjTexCoord.w)*0.5+0.5;projTexCoord=clamp(projTexCoord,0.002,0.998);float fragCoordZ=unpackRGBAToDepth(texture2D(depthBuffer,projTexCoord));\n#ifdef PERSPECTIVE_CAMERA\nfloat viewZ=perspectiveDepthToViewZ(fragCoordZ,cameraNear,cameraFar);\n#else\nfloat viewZ=orthographicDepthToViewZ(fragCoordZ,cameraNear,cameraFar);\n#endif\nfloat depthTest=(-vViewZ>-viewZ)? 1.0 : 0.0;gl_FragColor.rg=vec2(0.0,depthTest);}";

// src/materials/glsl/depth-comparison/shader.vert
var shader_default11 = "#include <common>\n#include <morphtarget_pars_vertex>\n#include <skinning_pars_vertex>\n#include <clipping_planes_pars_vertex>\nvarying float vViewZ;varying vec4 vProjTexCoord;void main(){\n#include <skinbase_vertex>\n#include <begin_vertex>\n#include <morphtarget_vertex>\n#include <skinning_vertex>\n#include <project_vertex>\nvViewZ=mvPosition.z;vProjTexCoord=gl_Position;\n#include <clipping_planes_vertex>\n}";

// src/materials/DepthComparisonMaterial.js
var DepthComparisonMaterial = class extends ShaderMaterial7 {
  constructor(depthTexture = null, camera) {
    super({
      type: "DepthComparisonMaterial",
      uniforms: {
        depthBuffer: new Uniform7(depthTexture),
        cameraNear: new Uniform7(0.3),
        cameraFar: new Uniform7(1e3)
      },
      fragmentShader: shader_default10,
      vertexShader: shader_default11,
      blending: NoBlending7,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.adoptCameraSettings(camera);
  }
  adoptCameraSettings(camera = null) {
    if (camera !== null) {
      this.uniforms.cameraNear.value = camera.near;
      this.uniforms.cameraFar.value = camera.far;
      if (camera instanceof PerspectiveCamera2) {
        this.defines.PERSPECTIVE_CAMERA = "1";
      } else {
        delete this.defines.PERSPECTIVE_CAMERA;
      }
    }
  }
};

// src/materials/DepthCopyMaterial.js
import {NoBlending as NoBlending8, ShaderMaterial as ShaderMaterial8, Uniform as Uniform8, Vector2 as Vector24} from "../build/three.module.js";

// src/materials/glsl/depth-copy/shader.frag
var shader_default12 = "#include <packing>\n#if INPUT_DEPTH_PACKING == 3201\nuniform lowp sampler2D depthBuffer;\n#else\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer;\n#else\nuniform mediump sampler2D depthBuffer;\n#endif\n#endif\nvarying vec2 vUv;void main(){\n#if INPUT_DEPTH_PACKING == OUTPUT_DEPTH_PACKING\ngl_FragColor=texture2D(depthBuffer,vUv);\n#else\n#if INPUT_DEPTH_PACKING == 3201\nfloat depth=unpackRGBAToDepth(texture2D(depthBuffer,vUv));gl_FragColor=vec4(vec3(depth),1.0);\n#else\nfloat depth=texture2D(depthBuffer,vUv).r;gl_FragColor=packDepthToRGBA(depth);\n#endif\n#endif\n}";

// src/materials/glsl/depth-copy/shader.vert
var shader_default13 = "varying vec2 vUv;\n#if DEPTH_COPY_MODE == 1\nuniform vec2 screenPosition;\n#endif\nvoid main(){\n#if DEPTH_COPY_MODE == 1\nvUv=screenPosition;\n#else\nvUv=position.xy*0.5+0.5;\n#endif\ngl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/DepthCopyMaterial.js
var DepthCopyMaterial = class extends ShaderMaterial8 {
  constructor() {
    super({
      type: "DepthCopyMaterial",
      defines: {
        INPUT_DEPTH_PACKING: "0",
        OUTPUT_DEPTH_PACKING: "0",
        DEPTH_COPY_MODE: "0"
      },
      uniforms: {
        depthBuffer: new Uniform8(null),
        screenPosition: new Uniform8(new Vector24())
      },
      fragmentShader: shader_default12,
      vertexShader: shader_default13,
      blending: NoBlending8,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.mode = DepthCopyMode.FULL;
  }
  getInputDepthPacking() {
    return Number(this.defines.INPUT_DEPTH_PACKING);
  }
  setInputDepthPacking(value) {
    this.defines.INPUT_DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  getOutputDepthPacking() {
    return Number(this.defines.OUTPUT_DEPTH_PACKING);
  }
  setOutputDepthPacking(value) {
    this.defines.OUTPUT_DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  getMode() {
    return this.mode;
  }
  setMode(value) {
    this.mode = value;
    this.defines.DEPTH_COPY_MODE = value.toFixed(0);
    this.needsUpdate = true;
  }
};
var DepthCopyMode = {
  FULL: 0,
  SINGLE: 1
};

// src/materials/DepthDownsamplingMaterial.js
import {NoBlending as NoBlending9, ShaderMaterial as ShaderMaterial9, Uniform as Uniform9, Vector2 as Vector25} from "../build/three.module.js";

// src/materials/glsl/depth-downsampling/shader.frag
var shader_default14 = "#include <packing>\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer;\n#else\nuniform mediump sampler2D depthBuffer;\n#endif\n#ifdef DOWNSAMPLE_NORMALS\nuniform lowp sampler2D normalBuffer;\n#endif\nvarying vec2 vUv0;varying vec2 vUv1;varying vec2 vUv2;varying vec2 vUv3;float readDepth(const in vec2 uv){\n#if DEPTH_PACKING == 3201\nreturn unpackRGBAToDepth(texture2D(depthBuffer,uv));\n#else\nreturn texture2D(depthBuffer,uv).r;\n#endif\n}int findBestDepth(const in float samples[4]){float c=(samples[0]+samples[1]+samples[2]+samples[3])/4.0;float distances[4];distances[0]=abs(c-samples[0]);distances[1]=abs(c-samples[1]);distances[2]=abs(c-samples[2]);distances[3]=abs(c-samples[3]);float maxDistance=max(max(distances[0],distances[1]),max(distances[2],distances[3]));int remaining[3];int rejected[3];int i,j,k;for(i=0,j=0,k=0;i<4;++i){if(distances[i]<maxDistance){remaining[j++]=i;}else{rejected[k++]=i;}}for(;j<3;++j){remaining[j]=rejected[--k];}vec3 s=vec3(samples[remaining[0]],samples[remaining[1]],samples[remaining[2]]);c=(s.x+s.y+s.z)/3.0;distances[0]=abs(c-s.x);distances[1]=abs(c-s.y);distances[2]=abs(c-s.z);float minDistance=min(distances[0],min(distances[1],distances[2]));for(i=0;i<3;++i){if(distances[i]==minDistance){break;}}return remaining[i];}void main(){float d[4];d[0]=readDepth(vUv0);d[1]=readDepth(vUv1);d[2]=readDepth(vUv2);d[3]=readDepth(vUv3);int index=findBestDepth(d);\n#ifdef DOWNSAMPLE_NORMALS\nvec2 uvs[4];uvs[0]=vUv0;uvs[1]=vUv1;uvs[2]=vUv2;uvs[3]=vUv3;vec3 n=texture2D(normalBuffer,uvs[index]).rgb;\n#else\nvec3 n=vec3(0.0);\n#endif\ngl_FragColor=vec4(n,d[index]);}";

// src/materials/glsl/depth-downsampling/shader.vert
var shader_default15 = "uniform vec2 texelSize;varying vec2 vUv0;varying vec2 vUv1;varying vec2 vUv2;varying vec2 vUv3;void main(){vec2 uv=position.xy*0.5+0.5;vUv0=uv;vUv1=vec2(uv.x,uv.y+texelSize.y);vUv2=vec2(uv.x+texelSize.x,uv.y);vUv3=uv+texelSize;gl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/DepthDownsamplingMaterial.js
var DepthDownsamplingMaterial = class extends ShaderMaterial9 {
  constructor() {
    super({
      type: "DepthDownsamplingMaterial",
      defines: {
        DEPTH_PACKING: "0"
      },
      uniforms: {
        depthBuffer: new Uniform9(null),
        normalBuffer: new Uniform9(null),
        texelSize: new Uniform9(new Vector25())
      },
      fragmentShader: shader_default14,
      vertexShader: shader_default15,
      blending: NoBlending9,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
  }
  get depthPacking() {
    return Number(this.defines.DEPTH_PACKING);
  }
  set depthPacking(value) {
    this.defines.DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  setTexelSize(x, y) {
    this.uniforms.texelSize.value.set(x, y);
  }
};

// src/materials/DepthMaskMaterial.js
import {
  AlwaysDepth,
  EqualDepth,
  GreaterDepth,
  GreaterEqualDepth,
  LessDepth,
  LessEqualDepth,
  NeverDepth,
  NoBlending as NoBlending10,
  NotEqualDepth,
  ShaderMaterial as ShaderMaterial10,
  Uniform as Uniform10
} from "../build/three.module.js";

// src/materials/glsl/depth-mask/shader.frag
var shader_default16 = "#include <common>\n#include <packing>\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer0;uniform highp sampler2D depthBuffer1;\n#else\nuniform mediump sampler2D depthBuffer0;uniform mediump sampler2D depthBuffer1;\n#endif\nuniform sampler2D inputBuffer;uniform float bias0;uniform float bias1;varying vec2 vUv;void main(){vec2 depth;\n#if DEPTH_PACKING_0 == 3201\ndepth.x=unpackRGBAToDepth(texture2D(depthBuffer0,vUv));\n#else\ndepth.x=texture2D(depthBuffer0,vUv).r;\n#endif\n#if DEPTH_PACKING_1 == 3201\ndepth.y=unpackRGBAToDepth(texture2D(depthBuffer1,vUv));\n#else\ndepth.y=texture2D(depthBuffer1,vUv).r;\n#endif\ndepth=clamp(depth+vec2(bias0,bias1),0.0,1.0);\n#ifdef KEEP_FAR\nbool keep=(depth.x==1.0)||depthTest(depth.x,depth.y);\n#else\nbool keep=(depth.x!=1.0)&&depthTest(depth.x,depth.y);\n#endif\nif(keep){gl_FragColor=texture2D(inputBuffer,vUv);}else{discard;}}";

// src/materials/DepthMaskMaterial.js
var DepthMaskMaterial = class extends ShaderMaterial10 {
  constructor() {
    super({
      type: "DepthMaskMaterial",
      defines: {
        DEPTH_EPSILON: "0.00001",
        DEPTH_PACKING_0: "0",
        DEPTH_PACKING_1: "0",
        KEEP_FAR: "1"
      },
      uniforms: {
        inputBuffer: new Uniform10(null),
        depthBuffer0: new Uniform10(null),
        depthBuffer1: new Uniform10(null),
        bias0: new Uniform10(0),
        bias1: new Uniform10(0)
      },
      fragmentShader: shader_default16,
      vertexShader: shader_default2,
      blending: NoBlending10,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.depthMode = LessDepth;
    this.setDepthMode(LessDepth);
  }
  get keepFar() {
    return this.defines.KEEP_FAR !== void 0;
  }
  set keepFar(value) {
    if (value) {
      this.defines.KEEP_FAR = "1";
    } else {
      delete this.defines.KEEP_FAR;
    }
    this.needsUpdate = true;
  }
  getEpsilon() {
    return Number(this.defines.DEPTH_EPSILON);
  }
  setEpsilon(value) {
    this.defines.DEPTH_EPSILON = value.toFixed(16);
    this.needsUpdate = true;
  }
  getDepthMode() {
    return this.depthMode;
  }
  setDepthMode(mode) {
    let depthTest;
    switch (mode) {
      case NeverDepth:
        depthTest = "false";
        break;
      case AlwaysDepth:
        depthTest = "true";
        break;
      case EqualDepth:
        depthTest = "abs(d1 - d0) <= DEPTH_EPSILON";
        break;
      case NotEqualDepth:
        depthTest = "abs(d1 - d0) > DEPTH_EPSILON";
        break;
      case LessDepth:
        depthTest = "d0 > d1";
        break;
      case LessEqualDepth:
        depthTest = "d0 >= d1";
        break;
      case GreaterEqualDepth:
        depthTest = "d0 <= d1";
        break;
      case GreaterDepth:
      default:
        depthTest = "d0 < d1";
        break;
    }
    this.depthMode = mode;
    this.defines["depthTest(d0, d1)"] = depthTest;
    this.needsUpdate = true;
  }
};

// src/materials/EdgeDetectionMaterial.js
import {NoBlending as NoBlending11, ShaderMaterial as ShaderMaterial11, Uniform as Uniform11, Vector2 as Vector26} from "../build/three.module.js";
var EdgeDetectionMaterial = class extends ShaderMaterial11 {
  constructor(texelSize = new Vector26(), mode = EdgeDetectionMode.COLOR) {
    super({
      type: "EdgeDetectionMaterial",
      defines: {
        LOCAL_CONTRAST_ADAPTATION_FACTOR: "2.0",
        EDGE_THRESHOLD: "0.1",
        DEPTH_THRESHOLD: "0.01",
        PREDICATION_MODE: "0",
        PREDICATION_THRESHOLD: "0.01",
        PREDICATION_SCALE: "2.0",
        PREDICATION_STRENGTH: "1.0",
        DEPTH_PACKING: "0"
      },
      uniforms: {
        inputBuffer: new Uniform11(null),
        depthBuffer: new Uniform11(null),
        predicationBuffer: new Uniform11(null),
        texelSize: new Uniform11(texelSize)
      },
      fragmentShader: shader_default5,
      vertexShader: shader_default6,
      blending: NoBlending11,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.setEdgeDetectionMode(mode);
  }
  get depthPacking() {
    return Number(this.defines.DEPTH_PACKING);
  }
  set depthPacking(value) {
    this.defines.DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  setEdgeDetectionMode(mode) {
    this.defines.EDGE_DETECTION_MODE = mode.toFixed(0);
    this.needsUpdate = true;
  }
  setLocalContrastAdaptationFactor(factor) {
    this.defines.LOCAL_CONTRAST_ADAPTATION_FACTOR = factor.toFixed("6");
    this.needsUpdate = true;
  }
  setEdgeDetectionThreshold(threshold) {
    this.defines.EDGE_THRESHOLD = threshold.toFixed("6");
    this.defines.DEPTH_THRESHOLD = (threshold * 0.1).toFixed("6");
    this.needsUpdate = true;
  }
  setPredicationMode(mode) {
    this.defines.PREDICATION_MODE = mode.toFixed(0);
    this.needsUpdate = true;
  }
  setPredicationBuffer(predicationBuffer) {
    this.uniforms.predicationBuffer.value = predicationBuffer;
  }
  setPredicationThreshold(threshold) {
    this.defines.PREDICATION_THRESHOLD = threshold.toFixed("6");
    this.needsUpdate = true;
  }
  setPredicationScale(scale) {
    this.defines.PREDICATION_SCALE = scale.toFixed("6");
    this.needsUpdate = true;
  }
  setPredicationStrength(strength) {
    this.defines.PREDICATION_STRENGTH = strength.toFixed("6");
    this.needsUpdate = true;
  }
};
var EdgeDetectionMode = {
  DEPTH: 0,
  LUMA: 1,
  COLOR: 2
};
var PredicationMode = {
  DISABLED: 0,
  DEPTH: 1,
  CUSTOM: 2
};

// src/materials/EffectMaterial.js
import {NoBlending as NoBlending12, PerspectiveCamera as PerspectiveCamera3, ShaderMaterial as ShaderMaterial12, Uniform as Uniform12, Vector2 as Vector27} from "../build/three.module.js";

// src/materials/glsl/effect/shader.frag
var shader_default17 = "#include <common>\n#include <packing>\n#include <dithering_pars_fragment>\n#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D depthBuffer;\n#else\nuniform mediump sampler2D depthBuffer;\n#endif\nuniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;float readDepth(const in vec2 uv){\n#if DEPTH_PACKING == 3201\nreturn unpackRGBAToDepth(texture2D(depthBuffer,uv));\n#else\nreturn texture2D(depthBuffer,uv).r;\n#endif\n}float getViewZ(const in float depth){\n#ifdef PERSPECTIVE_CAMERA\nreturn perspectiveDepthToViewZ(depth,cameraNear,cameraFar);\n#else\nreturn orthographicDepthToViewZ(depth,cameraNear,cameraFar);\n#endif\n}FRAGMENT_HEADvoid main(){FRAGMENT_MAIN_UVvec4 color0=texture2D(inputBuffer,UV);vec4 color1=vec4(0.0);FRAGMENT_MAIN_IMAGEgl_FragColor=color0;\n#ifdef ENCODE_OUTPUT\n#include <encodings_fragment>\n#endif\n#include <dithering_fragment>\n}";

// src/materials/glsl/effect/shader.vert
var shader_default18 = "uniform vec2 resolution;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float aspect;uniform float time;varying vec2 vUv;VERTEX_HEADvoid main(){vUv=position.xy*0.5+0.5;VERTEX_MAIN_SUPPORTgl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/EffectMaterial.js
var EffectMaterial = class extends ShaderMaterial12 {
  constructor(shaderParts = null, defines = null, uniforms = null, camera, dithering = false) {
    super({
      type: "EffectMaterial",
      defines: {
        DEPTH_PACKING: "0",
        ENCODE_OUTPUT: "1"
      },
      uniforms: {
        inputBuffer: new Uniform12(null),
        depthBuffer: new Uniform12(null),
        resolution: new Uniform12(new Vector27()),
        texelSize: new Uniform12(new Vector27()),
        cameraNear: new Uniform12(0.3),
        cameraFar: new Uniform12(1e3),
        aspect: new Uniform12(1),
        time: new Uniform12(0)
      },
      blending: NoBlending12,
      depthWrite: false,
      depthTest: false,
      dithering
    });
    this.toneMapped = false;
    if (shaderParts !== null) {
      this.setShaderParts(shaderParts);
    }
    if (defines !== null) {
      this.setDefines(defines);
    }
    if (uniforms !== null) {
      this.setUniforms(uniforms);
    }
    this.adoptCameraSettings(camera);
  }
  get depthPacking() {
    return Number(this.defines.DEPTH_PACKING);
  }
  set depthPacking(value) {
    this.defines.DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  setShaderParts(shaderParts) {
    this.fragmentShader = shader_default17.replace(Section.FRAGMENT_HEAD, shaderParts.get(Section.FRAGMENT_HEAD)).replace(Section.FRAGMENT_MAIN_UV, shaderParts.get(Section.FRAGMENT_MAIN_UV)).replace(Section.FRAGMENT_MAIN_IMAGE, shaderParts.get(Section.FRAGMENT_MAIN_IMAGE));
    this.vertexShader = shader_default18.replace(Section.VERTEX_HEAD, shaderParts.get(Section.VERTEX_HEAD)).replace(Section.VERTEX_MAIN_SUPPORT, shaderParts.get(Section.VERTEX_MAIN_SUPPORT));
    this.needsUpdate = true;
    return this;
  }
  setDefines(defines) {
    for (const entry of defines.entries()) {
      this.defines[entry[0]] = entry[1];
    }
    this.needsUpdate = true;
    return this;
  }
  setUniforms(uniforms) {
    for (const entry of uniforms.entries()) {
      this.uniforms[entry[0]] = entry[1];
    }
    return this;
  }
  adoptCameraSettings(camera = null) {
    if (camera !== null) {
      this.uniforms.cameraNear.value = camera.near;
      this.uniforms.cameraFar.value = camera.far;
      if (camera instanceof PerspectiveCamera3) {
        this.defines.PERSPECTIVE_CAMERA = "1";
      } else {
        delete this.defines.PERSPECTIVE_CAMERA;
      }
      this.needsUpdate = true;
    }
  }
  setSize(width, height) {
    const w = Math.max(width, 1);
    const h = Math.max(height, 1);
    this.uniforms.resolution.value.set(w, h);
    this.uniforms.texelSize.value.set(1 / w, 1 / h);
    this.uniforms.aspect.value = w / h;
  }
};
var Section = {
  FRAGMENT_HEAD: "FRAGMENT_HEAD",
  FRAGMENT_MAIN_UV: "FRAGMENT_MAIN_UV",
  FRAGMENT_MAIN_IMAGE: "FRAGMENT_MAIN_IMAGE",
  VERTEX_HEAD: "VERTEX_HEAD",
  VERTEX_MAIN_SUPPORT: "VERTEX_MAIN_SUPPORT"
};

// src/materials/GodRaysMaterial.js
import {NoBlending as NoBlending13, ShaderMaterial as ShaderMaterial13, Uniform as Uniform13} from "../build/three.module.js";

// src/materials/glsl/god-rays/shader.frag
var shader_default19 = "#include <common>\n#include <dithering_pars_fragment>\n#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\nuniform vec2 lightPosition;uniform float exposure;uniform float decay;uniform float density;uniform float weight;uniform float clampMax;varying vec2 vUv;void main(){vec2 coord=vUv;vec2 delta=lightPosition-coord;delta*=1.0/SAMPLES_FLOAT*density;float illuminationDecay=1.0;vec4 color=vec4(0.0);for(int i=0;i<SAMPLES_INT;++i){coord+=delta;vec4 texel=texture2D(inputBuffer,coord);texel*=illuminationDecay*weight;color+=texel;illuminationDecay*=decay;}gl_FragColor=clamp(color*exposure,0.0,clampMax);\n#include <dithering_fragment>\n}";

// src/materials/GodRaysMaterial.js
var GodRaysMaterial = class extends ShaderMaterial13 {
  constructor(lightPosition) {
    super({
      type: "GodRaysMaterial",
      defines: {
        SAMPLES_INT: "60",
        SAMPLES_FLOAT: "60.0"
      },
      uniforms: {
        inputBuffer: new Uniform13(null),
        lightPosition: new Uniform13(lightPosition),
        density: new Uniform13(1),
        decay: new Uniform13(1),
        weight: new Uniform13(1),
        exposure: new Uniform13(1),
        clampMax: new Uniform13(1)
      },
      fragmentShader: shader_default19,
      vertexShader: shader_default2,
      blending: NoBlending13,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
  }
  get samples() {
    return Number(this.defines.SAMPLES_INT);
  }
  set samples(value) {
    const s = Math.floor(value);
    this.defines.SAMPLES_INT = s.toFixed(0);
    this.defines.SAMPLES_FLOAT = s.toFixed(1);
    this.needsUpdate = true;
  }
};

// src/materials/LuminanceMaterial.js
import {NoBlending as NoBlending14, ShaderMaterial as ShaderMaterial14, Uniform as Uniform14, Vector2 as Vector28} from "../build/three.module.js";

// src/materials/glsl/luminance/shader.frag
var shader_default20 = "#include <common>\n#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\n#ifdef RANGE\nuniform vec2 range;\n#elif defined(THRESHOLD)\nuniform float threshold;uniform float smoothing;\n#endif\nvarying vec2 vUv;void main(){vec4 texel=texture2D(inputBuffer,vUv);float l=linearToRelativeLuminance(texel.rgb);\n#ifdef RANGE\nfloat low=step(range.x,l);float high=step(l,range.y);l*=low*high;\n#elif defined(THRESHOLD)\nl=smoothstep(threshold,threshold+smoothing,l);\n#endif\n#ifdef COLOR\ngl_FragColor=vec4(texel.rgb*l,l);\n#else\ngl_FragColor=vec4(l);\n#endif\n}";

// src/materials/LuminanceMaterial.js
var LuminanceMaterial = class extends ShaderMaterial14 {
  constructor(colorOutput = false, luminanceRange = null) {
    const useRange = luminanceRange !== null;
    super({
      type: "LuminanceMaterial",
      uniforms: {
        inputBuffer: new Uniform14(null),
        threshold: new Uniform14(0),
        smoothing: new Uniform14(1),
        range: new Uniform14(useRange ? luminanceRange : new Vector28())
      },
      fragmentShader: shader_default20,
      vertexShader: shader_default2,
      blending: NoBlending14,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.colorOutput = colorOutput;
    this.useThreshold = true;
    this.useRange = useRange;
  }
  get threshold() {
    return this.uniforms.threshold.value;
  }
  set threshold(value) {
    this.uniforms.threshold.value = value;
  }
  get smoothing() {
    return this.uniforms.smoothing.value;
  }
  set smoothing(value) {
    this.uniforms.smoothing.value = value;
  }
  get useThreshold() {
    return this.defines.THRESHOLD !== void 0;
  }
  set useThreshold(value) {
    if (value) {
      this.defines.THRESHOLD = "1";
    } else {
      delete this.defines.THRESHOLD;
    }
    this.needsUpdate = true;
  }
  get colorOutput() {
    return this.defines.COLOR !== void 0;
  }
  set colorOutput(value) {
    if (value) {
      this.defines.COLOR = "1";
    } else {
      delete this.defines.COLOR;
    }
    this.needsUpdate = true;
  }
  setColorOutputEnabled(enabled) {
    this.colorOutput = enabled;
  }
  get useRange() {
    return this.defines.RANGE !== void 0;
  }
  set useRange(value) {
    if (value) {
      this.defines.RANGE = "1";
    } else {
      delete this.defines.RANGE;
    }
    this.needsUpdate = true;
  }
  get luminanceRange() {
    return this.useRange;
  }
  set luminanceRange(value) {
    this.useRange = value;
  }
  setLuminanceRangeEnabled(enabled) {
    this.useRange = enabled;
  }
};

// src/materials/MaskMaterial.js
import {NoBlending as NoBlending15, ShaderMaterial as ShaderMaterial15, Uniform as Uniform15, UnsignedByteType} from "../build/three.module.js";

// src/materials/glsl/mask/shader.frag
var shader_default21 = "#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\n#ifdef MASK_PRECISION_HIGH\nuniform mediump sampler2D maskTexture;\n#else\nuniform lowp sampler2D maskTexture;\n#endif\n#if MASK_FUNCTION != 0\nuniform float strength;\n#endif\nvarying vec2 vUv;void main(){\n#if COLOR_CHANNEL == 0\nfloat mask=texture2D(maskTexture,vUv).r;\n#elif COLOR_CHANNEL == 1\nfloat mask=texture2D(maskTexture,vUv).g;\n#elif COLOR_CHANNEL == 2\nfloat mask=texture2D(maskTexture,vUv).b;\n#else\nfloat mask=texture2D(maskTexture,vUv).a;\n#endif\n#if MASK_FUNCTION == 0\n#ifdef INVERTED\nif(mask>0.0){discard;}\n#else\nif(mask==0.0){discard;}\n#endif\n#else\nmask=clamp(mask*strength,0.0,1.0);\n#ifdef INVERTED\nmask=(1.0-mask);\n#endif\n#if MASK_FUNCTION == 1\ngl_FragColor=mask*texture2D(inputBuffer,vUv);\n#else\ngl_FragColor=vec4(mask*texture2D(inputBuffer,vUv).rgb,mask);\n#endif\n#endif\n}";

// src/materials/MaskMaterial.js
var MaskMaterial = class extends ShaderMaterial15 {
  constructor(maskTexture = null) {
    super({
      type: "MaskMaterial",
      uniforms: {
        maskTexture: new Uniform15(maskTexture),
        inputBuffer: new Uniform15(null),
        strength: new Uniform15(1)
      },
      fragmentShader: shader_default21,
      vertexShader: shader_default2,
      blending: NoBlending15,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.colorChannel = ColorChannel.RED;
    this.maskFunction = MaskFunction.DISCARD;
  }
  set maskTexture(value) {
    this.uniforms.maskTexture.value = value;
    delete this.defines.MASK_PRECISION_HIGH;
    if (value.type !== UnsignedByteType) {
      this.defines.MASK_PRECISION_HIGH = "1";
    }
    this.needsUpdate = true;
  }
  set colorChannel(value) {
    this.defines.COLOR_CHANNEL = value.toFixed(0);
    this.needsUpdate = true;
  }
  set maskFunction(value) {
    this.defines.MASK_FUNCTION = value.toFixed(0);
    this.needsUpdate = true;
  }
  get inverted() {
    return this.defines.INVERTED !== void 0;
  }
  set inverted(value) {
    if (this.inverted && !value) {
      delete this.defines.INVERTED;
    } else if (value) {
      this.defines.INVERTED = "1";
    }
    this.needsUpdate = true;
  }
  get strength() {
    return this.uniforms.strength.value;
  }
  set strength(value) {
    this.uniforms.strength.value = value;
  }
};
var MaskFunction = {
  DISCARD: 0,
  MULTIPLY: 1,
  MULTIPLY_RGB_SET_ALPHA: 2
};

// src/materials/OutlineMaterial.js
import {NoBlending as NoBlending16, ShaderMaterial as ShaderMaterial16, Uniform as Uniform16, Vector2 as Vector29} from "../build/three.module.js";

// src/materials/glsl/outline/shader.frag
var shader_default22 = "uniform lowp sampler2D inputBuffer;varying vec2 vUv0;varying vec2 vUv1;varying vec2 vUv2;varying vec2 vUv3;void main(){vec2 c0=texture2D(inputBuffer,vUv0).rg;vec2 c1=texture2D(inputBuffer,vUv1).rg;vec2 c2=texture2D(inputBuffer,vUv2).rg;vec2 c3=texture2D(inputBuffer,vUv3).rg;float d0=(c0.x-c1.x)*0.5;float d1=(c2.x-c3.x)*0.5;float d=length(vec2(d0,d1));float a0=min(c0.y,c1.y);float a1=min(c2.y,c3.y);float visibilityFactor=min(a0,a1);gl_FragColor.rg=(1.0-visibilityFactor>0.001)? vec2(d,0.0): vec2(0.0,d);}";

// src/materials/glsl/outline/shader.vert
var shader_default23 = "uniform vec2 texelSize;varying vec2 vUv0;varying vec2 vUv1;varying vec2 vUv2;varying vec2 vUv3;void main(){vec2 uv=position.xy*0.5+0.5;vUv0=vec2(uv.x+texelSize.x,uv.y);vUv1=vec2(uv.x-texelSize.x,uv.y);vUv2=vec2(uv.x,uv.y+texelSize.y);vUv3=vec2(uv.x,uv.y-texelSize.y);gl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/OutlineMaterial.js
var OutlineMaterial = class extends ShaderMaterial16 {
  constructor(texelSize = new Vector29()) {
    super({
      type: "OutlineMaterial",
      uniforms: {
        inputBuffer: new Uniform16(null),
        texelSize: new Uniform16(new Vector29())
      },
      fragmentShader: shader_default22,
      vertexShader: shader_default23,
      blending: NoBlending16,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.setTexelSize(texelSize.x, texelSize.y);
    this.uniforms.maskTexture = this.uniforms.inputBuffer;
  }
  setTexelSize(x, y) {
    this.uniforms.texelSize.value.set(x, y);
  }
};
var OutlineEdgesMaterial = OutlineMaterial;

// src/materials/SMAAWeightsMaterial.js
import {NoBlending as NoBlending17, ShaderMaterial as ShaderMaterial17, Uniform as Uniform17, Vector2 as Vector210} from "../build/three.module.js";

// src/materials/glsl/smaa-weights/shader.frag
var shader_default24 = "#define sampleLevelZeroOffset(t, coord, offset) texture2D(t, coord + offset * texelSize)\n#if __VERSION__ < 300\n#define round(v) floor(v + 0.5)\n#endif\n#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D inputBuffer;\n#else\nuniform lowp sampler2D inputBuffer;\n#endif\nuniform lowp sampler2D areaTexture;uniform lowp sampler2D searchTexture;uniform vec2 texelSize;uniform vec2 resolution;varying vec2 vUv;varying vec4 vOffset[3];varying vec2 vPixCoord;void movec(const in bvec2 c,inout vec2 variable,const in vec2 value){if(c.x){variable.x=value.x;}if(c.y){variable.y=value.y;}}void movec(const in bvec4 c,inout vec4 variable,const in vec4 value){movec(c.xy,variable.xy,value.xy);movec(c.zw,variable.zw,value.zw);}vec2 decodeDiagBilinearAccess(in vec2 e){e.r=e.r*abs(5.0*e.r-5.0*0.75);return round(e);}vec4 decodeDiagBilinearAccess(in vec4 e){e.rb=e.rb*abs(5.0*e.rb-5.0*0.75);return round(e);}vec2 searchDiag1(const in vec2 texCoord,const in vec2 dir,out vec2 e){vec4 coord=vec4(texCoord,-1.0,1.0);vec3 t=vec3(texelSize,1.0);for(int i=0;i<MAX_SEARCH_STEPS_INT;++i){if(!(coord.z<float(MAX_SEARCH_STEPS_DIAG_INT-1)&&coord.w>0.9)){break;}coord.xyz=t*vec3(dir,1.0)+coord.xyz;e=texture2D(inputBuffer,coord.xy).rg;coord.w=dot(e,vec2(0.5));}return coord.zw;}vec2 searchDiag2(const in vec2 texCoord,const in vec2 dir,out vec2 e){vec4 coord=vec4(texCoord,-1.0,1.0);coord.x+=0.25*texelSize.x;vec3 t=vec3(texelSize,1.0);for(int i=0;i<MAX_SEARCH_STEPS_INT;++i){if(!(coord.z<float(MAX_SEARCH_STEPS_DIAG_INT-1)&&coord.w>0.9)){break;}coord.xyz=t*vec3(dir,1.0)+coord.xyz;e=texture2D(inputBuffer,coord.xy).rg;e=decodeDiagBilinearAccess(e);coord.w=dot(e,vec2(0.5));}return coord.zw;}vec2 areaDiag(const in vec2 dist,const in vec2 e,const in float offset){vec2 texCoord=vec2(AREATEX_MAX_DISTANCE_DIAG,AREATEX_MAX_DISTANCE_DIAG)*e+dist;texCoord=AREATEX_PIXEL_SIZE*texCoord+0.5*AREATEX_PIXEL_SIZE;texCoord.x+=0.5;texCoord.y+=AREATEX_SUBTEX_SIZE*offset;return texture2D(areaTexture,texCoord).rg;}vec2 calculateDiagWeights(const in vec2 texCoord,const in vec2 e,const in vec4 subsampleIndices){vec2 weights=vec2(0.0);vec4 d;vec2 end;if(e.r>0.0){d.xz=searchDiag1(texCoord,vec2(-1.0,1.0),end);d.x+=float(end.y>0.9);}else{d.xz=vec2(0.0);}d.yw=searchDiag1(texCoord,vec2(1.0,-1.0),end);if(d.x+d.y>2.0){vec4 coords=vec4(-d.x+0.25,d.x,d.y,-d.y-0.25)*texelSize.xyxy+texCoord.xyxy;vec4 c;c.xy=sampleLevelZeroOffset(inputBuffer,coords.xy,vec2(-1,0)).rg;c.zw=sampleLevelZeroOffset(inputBuffer,coords.zw,vec2(1,0)).rg;c.yxwz=decodeDiagBilinearAccess(c.xyzw);vec2 cc=vec2(2.0)*c.xz+c.yw;movec(bvec2(step(0.9,d.zw)),cc,vec2(0.0));weights+=areaDiag(d.xy,cc,subsampleIndices.z);}d.xz=searchDiag2(texCoord,vec2(-1.0,-1.0),end);if(sampleLevelZeroOffset(inputBuffer,texCoord,vec2(1,0)).r>0.0){d.yw=searchDiag2(texCoord,vec2(1.0),end);d.y+=float(end.y>0.9);}else{d.yw=vec2(0.0);}if(d.x+d.y>2.0){vec4 coords=vec4(-d.x,-d.x,d.y,d.y)*texelSize.xyxy+texCoord.xyxy;vec4 c;c.x=sampleLevelZeroOffset(inputBuffer,coords.xy,vec2(-1,0)).g;c.y=sampleLevelZeroOffset(inputBuffer,coords.xy,vec2(0,-1)).r;c.zw=sampleLevelZeroOffset(inputBuffer,coords.zw,vec2(1,0)).gr;vec2 cc=vec2(2.0)*c.xz+c.yw;movec(bvec2(step(0.9,d.zw)),cc,vec2(0.0));weights+=areaDiag(d.xy,cc,subsampleIndices.w).gr;}return weights;}float searchLength(const in vec2 e,const in float offset){vec2 scale=SEARCHTEX_SIZE*vec2(0.5,-1.0);vec2 bias=SEARCHTEX_SIZE*vec2(offset,1.0);scale+=vec2(-1.0,1.0);bias+=vec2(0.5,-0.5);scale*=1.0/SEARCHTEX_PACKED_SIZE;bias*=1.0/SEARCHTEX_PACKED_SIZE;return texture2D(searchTexture,scale*e+bias).r;}float searchXLeft(in vec2 texCoord,const in float end){vec2 e=vec2(0.0,1.0);for(int i=0;i<MAX_SEARCH_STEPS_INT;++i){if(!(texCoord.x>end&&e.g>0.8281&&e.r==0.0)){break;}e=texture2D(inputBuffer,texCoord).rg;texCoord=vec2(-2.0,0.0)*texelSize+texCoord;}float offset=-(255.0/127.0)*searchLength(e,0.0)+3.25;return texelSize.x*offset+texCoord.x;}float searchXRight(vec2 texCoord,const in float end){vec2 e=vec2(0.0,1.0);for(int i=0;i<MAX_SEARCH_STEPS_INT;++i){if(!(texCoord.x<end&&e.g>0.8281&&e.r==0.0)){break;}e=texture2D(inputBuffer,texCoord).rg;texCoord=vec2(2.0,0.0)*texelSize.xy+texCoord;}float offset=-(255.0/127.0)*searchLength(e,0.5)+3.25;return-texelSize.x*offset+texCoord.x;}float searchYUp(vec2 texCoord,const in float end){vec2 e=vec2(1.0,0.0);for(int i=0;i<MAX_SEARCH_STEPS_INT;++i){if(!(texCoord.y>end&&e.r>0.8281&&e.g==0.0)){break;}e=texture2D(inputBuffer,texCoord).rg;texCoord=-vec2(0.0,2.0)*texelSize.xy+texCoord;}float offset=-(255.0/127.0)*searchLength(e.gr,0.0)+3.25;return texelSize.y*offset+texCoord.y;}float searchYDown(vec2 texCoord,const in float end){vec2 e=vec2(1.0,0.0);for(int i=0;i<MAX_SEARCH_STEPS_INT;i++){if(!(texCoord.y<end&&e.r>0.8281&&e.g==0.0)){break;}e=texture2D(inputBuffer,texCoord).rg;texCoord=vec2(0.0,2.0)*texelSize.xy+texCoord;}float offset=-(255.0/127.0)*searchLength(e.gr,0.5)+3.25;return-texelSize.y*offset+texCoord.y;}vec2 area(const in vec2 dist,const in float e1,const in float e2,const in float offset){vec2 texCoord=vec2(AREATEX_MAX_DISTANCE)*round(4.0*vec2(e1,e2))+dist;texCoord=AREATEX_PIXEL_SIZE*texCoord+0.5*AREATEX_PIXEL_SIZE;texCoord.y=AREATEX_SUBTEX_SIZE*offset+texCoord.y;return texture2D(areaTexture,texCoord).rg;}void detectHorizontalCornerPattern(inout vec2 weights,const in vec4 texCoord,const in vec2 d){\n#if !defined(DISABLE_CORNER_DETECTION)\nvec2 leftRight=step(d.xy,d.yx);vec2 rounding=(1.0-CORNER_ROUNDING_NORM)*leftRight;rounding/=leftRight.x+leftRight.y;vec2 factor=vec2(1.0);factor.x-=rounding.x*sampleLevelZeroOffset(inputBuffer,texCoord.xy,vec2(0,1)).r;factor.x-=rounding.y*sampleLevelZeroOffset(inputBuffer,texCoord.zw,vec2(1,1)).r;factor.y-=rounding.x*sampleLevelZeroOffset(inputBuffer,texCoord.xy,vec2(0,-2)).r;factor.y-=rounding.y*sampleLevelZeroOffset(inputBuffer,texCoord.zw,vec2(1,-2)).r;weights*=clamp(factor,0.0,1.0);\n#endif\n}void detectVerticalCornerPattern(inout vec2 weights,const in vec4 texCoord,const in vec2 d){\n#if !defined(DISABLE_CORNER_DETECTION)\nvec2 leftRight=step(d.xy,d.yx);vec2 rounding=(1.0-CORNER_ROUNDING_NORM)*leftRight;rounding/=leftRight.x+leftRight.y;vec2 factor=vec2(1.0);factor.x-=rounding.x*sampleLevelZeroOffset(inputBuffer,texCoord.xy,vec2(1,0)).g;factor.x-=rounding.y*sampleLevelZeroOffset(inputBuffer,texCoord.zw,vec2(1,1)).g;factor.y-=rounding.x*sampleLevelZeroOffset(inputBuffer,texCoord.xy,vec2(-2,0)).g;factor.y-=rounding.y*sampleLevelZeroOffset(inputBuffer,texCoord.zw,vec2(-2,1)).g;weights*=clamp(factor,0.0,1.0);\n#endif\n}void main(){vec4 weights=vec4(0.0);vec4 subsampleIndices=vec4(0.0);vec2 e=texture2D(inputBuffer,vUv).rg;if(e.g>0.0){\n#if !defined(DISABLE_DIAG_DETECTION)\nweights.rg=calculateDiagWeights(vUv,e,subsampleIndices);if(weights.r==-weights.g){\n#endif\nvec2 d;vec3 coords;coords.x=searchXLeft(vOffset[0].xy,vOffset[2].x);coords.y=vOffset[1].y;d.x=coords.x;float e1=texture2D(inputBuffer,coords.xy).r;coords.z=searchXRight(vOffset[0].zw,vOffset[2].y);d.y=coords.z;d=round(resolution.xx*d+-vPixCoord.xx);vec2 sqrtD=sqrt(abs(d));float e2=sampleLevelZeroOffset(inputBuffer,coords.zy,vec2(1,0)).r;weights.rg=area(sqrtD,e1,e2,subsampleIndices.y);coords.y=vUv.y;detectHorizontalCornerPattern(weights.rg,coords.xyzy,d);\n#if !defined(DISABLE_DIAG_DETECTION)\n}else{e.r=0.0;}\n#endif\n}if(e.r>0.0){vec2 d;vec3 coords;coords.y=searchYUp(vOffset[1].xy,vOffset[2].z);coords.x=vOffset[0].x;d.x=coords.y;float e1=texture2D(inputBuffer,coords.xy).g;coords.z=searchYDown(vOffset[1].zw,vOffset[2].w);d.y=coords.z;d=round(resolution.yy*d-vPixCoord.yy);vec2 sqrtD=sqrt(abs(d));float e2=sampleLevelZeroOffset(inputBuffer,coords.xz,vec2(0,1)).g;weights.ba=area(sqrtD,e1,e2,subsampleIndices.x);coords.x=vUv.x;detectVerticalCornerPattern(weights.ba,coords.xyxz,d);}gl_FragColor=weights;}";

// src/materials/glsl/smaa-weights/shader.vert
var shader_default25 = "uniform vec2 texelSize;uniform vec2 resolution;varying vec2 vUv;varying vec4 vOffset[3];varying vec2 vPixCoord;void main(){vUv=position.xy*0.5+0.5;vPixCoord=vUv*resolution;vOffset[0]=vUv.xyxy+texelSize.xyxy*vec4(-0.25,-0.125,1.25,-0.125);vOffset[1]=vUv.xyxy+texelSize.xyxy*vec4(-0.125,-0.25,-0.125,1.25);vOffset[2]=vec4(vOffset[0].xz,vOffset[1].yw)+vec4(-2.0,2.0,-2.0,2.0)*texelSize.xxyy*MAX_SEARCH_STEPS_FLOAT;gl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/SMAAWeightsMaterial.js
var SMAAWeightsMaterial = class extends ShaderMaterial17 {
  constructor(texelSize = new Vector210(), resolution = new Vector210()) {
    super({
      type: "SMAAWeightsMaterial",
      defines: {
        MAX_SEARCH_STEPS_INT: "16",
        MAX_SEARCH_STEPS_FLOAT: "16.0",
        MAX_SEARCH_STEPS_DIAG_INT: "8",
        MAX_SEARCH_STEPS_DIAG_FLOAT: "8.0",
        CORNER_ROUNDING: "25",
        CORNER_ROUNDING_NORM: "0.25",
        AREATEX_MAX_DISTANCE: "16.0",
        AREATEX_MAX_DISTANCE_DIAG: "20.0",
        AREATEX_PIXEL_SIZE: "(1.0 / vec2(160.0, 560.0))",
        AREATEX_SUBTEX_SIZE: "(1.0 / 7.0)",
        SEARCHTEX_SIZE: "vec2(66.0, 33.0)",
        SEARCHTEX_PACKED_SIZE: "vec2(64.0, 16.0)"
      },
      uniforms: {
        inputBuffer: new Uniform17(null),
        areaTexture: new Uniform17(null),
        searchTexture: new Uniform17(null),
        texelSize: new Uniform17(texelSize),
        resolution: new Uniform17(resolution)
      },
      fragmentShader: shader_default24,
      vertexShader: shader_default25,
      blending: NoBlending17,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
  }
  setOrthogonalSearchSteps(steps) {
    const s = Math.min(Math.max(steps, 0), 112);
    this.defines.MAX_SEARCH_STEPS_INT = s.toFixed("0");
    this.defines.MAX_SEARCH_STEPS_FLOAT = s.toFixed("1");
    this.needsUpdate = true;
  }
  setDiagonalSearchSteps(steps) {
    const s = Math.min(Math.max(steps, 0), 20);
    this.defines.MAX_SEARCH_STEPS_DIAG_INT = s.toFixed("0");
    this.defines.MAX_SEARCH_STEPS_DIAG_FLOAT = s.toFixed("1");
    this.needsUpdate = true;
  }
  setCornerRounding(rounding) {
    const r = Math.min(Math.max(rounding, 0), 100);
    this.defines.CORNER_ROUNDING = r.toFixed("4");
    this.defines.CORNER_ROUNDING_NORM = (r / 100).toFixed("4");
    this.needsUpdate = true;
  }
  get diagonalDetection() {
    return this.defines.DISABLE_DIAG_DETECTION === void 0;
  }
  set diagonalDetection(value) {
    if (value) {
      delete this.defines.DISABLE_DIAG_DETECTION;
    } else {
      this.defines.DISABLE_DIAG_DETECTION = "1";
    }
    this.needsUpdate = true;
  }
  get cornerRounding() {
    return this.defines.DISABLE_CORNER_DETECTION === void 0;
  }
  set cornerRounding(value) {
    if (value) {
      delete this.defines.DISABLE_CORNER_DETECTION;
    } else {
      this.defines.DISABLE_CORNER_DETECTION = "1";
    }
    this.needsUpdate = true;
  }
};

// src/materials/SSAOMaterial.js
import {
  Matrix4,
  NoBlending as NoBlending18,
  PerspectiveCamera as PerspectiveCamera4,
  ShaderMaterial as ShaderMaterial18,
  Uniform as Uniform18,
  Vector2 as Vector211
} from "../build/three.module.js";

// src/materials/glsl/ssao/shader.frag
var shader_default26 = "#include <common>\n#include <packing>\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D normalDepthBuffer;\n#else\nuniform mediump sampler2D normalDepthBuffer;\n#endif\n#ifndef NORMAL_DEPTH\nuniform lowp sampler2D normalBuffer;float readDepth(const in vec2 uv){\n#if DEPTH_PACKING == 3201\nreturn unpackRGBAToDepth(texture2D(normalDepthBuffer,uv));\n#else\nreturn texture2D(normalDepthBuffer,uv).r;\n#endif\n}\n#endif\nuniform lowp sampler2D noiseTexture;uniform mat4 inverseProjectionMatrix;uniform mat4 projectionMatrix;uniform vec2 texelSize;uniform float cameraNear;uniform float cameraFar;uniform float minRadiusScale;uniform float intensity;uniform float fade;uniform float bias;uniform vec2 distanceCutoff;uniform vec2 proximityCutoff;varying vec2 vUv;varying vec2 vUv2;float getViewZ(const in float depth){\n#ifdef PERSPECTIVE_CAMERA\nreturn perspectiveDepthToViewZ(depth,cameraNear,cameraFar);\n#else\nreturn orthographicDepthToViewZ(depth,cameraNear,cameraFar);\n#endif\n}vec3 getViewPosition(const in vec2 screenPosition,const in float depth,const in float viewZ){vec4 clipPosition=vec4(vec3(screenPosition,depth)*2.0-1.0,1.0);float clipW=projectionMatrix[2][3]*viewZ+projectionMatrix[3][3];clipPosition*=clipW;return(inverseProjectionMatrix*clipPosition).xyz;}float getAmbientOcclusion(const in vec3 p,const in vec3 n,const in float depth,const in vec2 uv){\n#ifdef DISTANCE_SCALING\nfloat radiusScale=1.0-smoothstep(0.0,distanceCutoff.y,depth);radiusScale=radiusScale*(1.0-minRadiusScale)+minRadiusScale;float radius=RADIUS*radiusScale;\n#else\nfloat radius=RADIUS;\n#endif\nfloat noise=texture2D(noiseTexture,vUv2).r;float baseAngle=noise*PI2;float invSamples=1.0/SAMPLES_FLOAT;float rings=SPIRAL_TURNS*PI2;float occlusion=0.0;int taps=0;for(int i=0;i<SAMPLES_INT;++i){float alpha=(float(i)+0.5)*invSamples;float angle=alpha*rings+baseAngle;vec2 coord=alpha*radius*vec2(cos(angle),sin(angle))*texelSize+uv;if(coord.s<0.0||coord.s>1.0||coord.t<0.0||coord.t>1.0){continue;}\n#ifdef NORMAL_DEPTH\nfloat sampleDepth=texture2D(normalDepthBuffer,coord).a;\n#else\nfloat sampleDepth=readDepth(coord);\n#endif\nfloat viewZ=getViewZ(sampleDepth);\n#ifdef PERSPECTIVE_CAMERA\nfloat linearSampleDepth=viewZToOrthographicDepth(viewZ,cameraNear,cameraFar);\n#else\nfloat linearSampleDepth=sampleDepth;\n#endif\nfloat proximity=abs(depth-linearSampleDepth);if(proximity<proximityCutoff.y){float falloff=1.0-smoothstep(proximityCutoff.x,proximityCutoff.y,proximity);vec3 Q=getViewPosition(coord,sampleDepth,viewZ);vec3 v=Q-p;float vv=dot(v,v);float vn=dot(v,n)-bias;float f=max(RADIUS_SQ-vv,0.0)/RADIUS_SQ;occlusion+=(f*f*f*max(vn/(fade+vv),0.0))*falloff;}++taps;}return occlusion/(4.0*max(float(taps),1.0));}void main(){\n#ifdef NORMAL_DEPTH\nvec4 normalDepth=texture2D(normalDepthBuffer,vUv);\n#else\nvec4 normalDepth=vec4(texture2D(normalBuffer,vUv).rgb,readDepth(vUv));\n#endif\nfloat ao=1.0;float depth=normalDepth.a;float viewZ=getViewZ(depth);\n#ifdef PERSPECTIVE_CAMERA\nfloat linearDepth=viewZToOrthographicDepth(viewZ,cameraNear,cameraFar);\n#else\nfloat linearDepth=depth;\n#endif\nif(linearDepth<distanceCutoff.y){vec3 viewPosition=getViewPosition(vUv,depth,viewZ);vec3 viewNormal=unpackRGBToNormal(normalDepth.rgb);ao-=getAmbientOcclusion(viewPosition,viewNormal,linearDepth,vUv);float d=smoothstep(distanceCutoff.x,distanceCutoff.y,linearDepth);ao=mix(ao,1.0,d);ao=clamp(pow(ao,abs(intensity)),0.0,1.0);}gl_FragColor.r=ao;}";

// src/materials/glsl/ssao/shader.vert
var shader_default27 = "uniform vec2 noiseScale;varying vec2 vUv;varying vec2 vUv2;void main(){vUv=position.xy*0.5+0.5;vUv2=vUv*noiseScale;gl_Position=vec4(position.xy,1.0,1.0);}";

// src/materials/SSAOMaterial.js
var SSAOMaterial = class extends ShaderMaterial18 {
  constructor(camera) {
    super({
      type: "SSAOMaterial",
      defines: {
        SAMPLES_INT: "0",
        SAMPLES_FLOAT: "0.0",
        SPIRAL_TURNS: "0.0",
        RADIUS: "1.0",
        RADIUS_SQ: "1.0",
        DISTANCE_SCALING: "1",
        DEPTH_PACKING: "0"
      },
      uniforms: {
        normalBuffer: new Uniform18(null),
        normalDepthBuffer: new Uniform18(null),
        noiseTexture: new Uniform18(null),
        inverseProjectionMatrix: new Uniform18(new Matrix4()),
        projectionMatrix: new Uniform18(new Matrix4()),
        texelSize: new Uniform18(new Vector211()),
        cameraNear: new Uniform18(0),
        cameraFar: new Uniform18(0),
        distanceCutoff: new Uniform18(new Vector211()),
        proximityCutoff: new Uniform18(new Vector211()),
        noiseScale: new Uniform18(new Vector211()),
        minRadiusScale: new Uniform18(0.33),
        intensity: new Uniform18(1),
        fade: new Uniform18(0.01),
        bias: new Uniform18(0)
      },
      fragmentShader: shader_default26,
      vertexShader: shader_default27,
      blending: NoBlending18,
      depthWrite: false,
      depthTest: false
    });
    this.toneMapped = false;
    this.adoptCameraSettings(camera);
  }
  get depthPacking() {
    return Number(this.defines.DEPTH_PACKING);
  }
  set depthPacking(value) {
    this.defines.DEPTH_PACKING = value.toFixed(0);
    this.needsUpdate = true;
  }
  setTexelSize(x, y) {
    this.uniforms.texelSize.value.set(x, y);
  }
  adoptCameraSettings(camera = null) {
    if (camera !== null) {
      const uniforms = this.uniforms;
      uniforms.cameraNear.value = camera.near;
      uniforms.cameraFar.value = camera.far;
      if (camera instanceof PerspectiveCamera4) {
        this.defines.PERSPECTIVE_CAMERA = "1";
      } else {
        delete this.defines.PERSPECTIVE_CAMERA;
      }
      this.needsUpdate = true;
    }
  }
};

// src/passes/Pass.js
import {
  BufferAttribute,
  BufferGeometry,
  Camera,
  Mesh,
  Scene
} from "../build/three.module.js";
var dummyCamera = new Camera();
var geometry = null;
function getFullscreenTriangle() {
  if (geometry === null) {
    const vertices = new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]);
    const uvs = new Float32Array([0, 0, 2, 0, 0, 2]);
    geometry = new BufferGeometry();
    if (geometry.setAttribute !== void 0) {
      geometry.setAttribute("position", new BufferAttribute(vertices, 3));
      geometry.setAttribute("uv", new BufferAttribute(uvs, 2));
    } else {
      geometry.addAttribute("position", new BufferAttribute(vertices, 3));
      geometry.addAttribute("uv", new BufferAttribute(uvs, 2));
    }
  }
  return geometry;
}
var Pass = class {
  constructor(name = "Pass", scene = new Scene(), camera = dummyCamera) {
    this.name = name;
    this.scene = scene;
    this.camera = camera;
    this.screen = null;
    this.rtt = true;
    this.needsSwap = true;
    this.needsDepthTexture = false;
    this.enabled = true;
  }
  get renderToScreen() {
    return !this.rtt;
  }
  set renderToScreen(value) {
    if (this.rtt === value) {
      const material = this.getFullscreenMaterial();
      if (material !== null) {
        material.needsUpdate = true;
      }
      this.rtt = !value;
    }
  }
  getFullscreenMaterial() {
    return this.screen !== null ? this.screen.material : null;
  }
  setFullscreenMaterial(material) {
    let screen = this.screen;
    if (screen !== null) {
      screen.material = material;
    } else {
      screen = new Mesh(getFullscreenTriangle(), material);
      screen.frustumCulled = false;
      if (this.scene === null) {
        this.scene = new Scene();
      }
      this.scene.add(screen);
      this.screen = screen;
    }
  }
  getDepthTexture() {
    return null;
  }
  setDepthTexture(depthTexture, depthPacking = 0) {
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    throw new Error("Render method not implemented!");
  }
  setSize(width, height) {
  }
  initialize(renderer, alpha, frameBufferType) {
  }
  dispose() {
    const material = this.getFullscreenMaterial();
    if (material !== null) {
      material.dispose();
    }
    for (const key of Object.keys(this)) {
      const property = this[key];
      if (property !== null && typeof property.dispose === "function") {
        if (property instanceof Scene) {
          continue;
        }
        this[key].dispose();
      }
    }
  }
};

// src/passes/SavePass.js
import {
  LinearFilter,
  RGBFormat,
  UnsignedByteType as UnsignedByteType2,
  WebGLRenderTarget
} from "../build/three.module.js";
var SavePass = class extends Pass {
  constructor(renderTarget, resize = true) {
    super("SavePass");
    this.setFullscreenMaterial(new CopyMaterial());
    this.needsSwap = false;
    this.renderTarget = renderTarget;
    if (renderTarget === void 0) {
      this.renderTarget = new WebGLRenderTarget(1, 1, {
        minFilter: LinearFilter,
        magFilter: LinearFilter,
        stencilBuffer: false,
        depthBuffer: false
      });
      this.renderTarget.texture.name = "SavePass.Target";
    }
    this.resize = resize;
  }
  get texture() {
    return this.renderTarget.texture;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    this.getFullscreenMaterial().uniforms.inputBuffer.value = inputBuffer.texture;
    renderer.setRenderTarget(this.renderToScreen ? null : this.renderTarget);
    renderer.render(this.scene, this.camera);
  }
  setSize(width, height) {
    if (this.resize) {
      const w = Math.max(width, 1);
      const h = Math.max(height, 1);
      this.renderTarget.setSize(w, h);
    }
  }
  initialize(renderer, alpha, frameBufferType) {
    if (!alpha && frameBufferType === UnsignedByteType2) {
      this.renderTarget.texture.format = RGBFormat;
    }
    if (frameBufferType !== void 0) {
      this.renderTarget.texture.type = frameBufferType;
      if (frameBufferType !== UnsignedByteType2) {
        const material = this.getFullscreenMaterial();
        material.defines.FRAMEBUFFER_PRECISION_HIGH = "1";
      }
    }
  }
};

// src/passes/AdaptiveLuminancePass.js
var AdaptiveLuminancePass = class extends Pass {
  constructor(luminanceBuffer, {
    minLuminance = 0.01,
    adaptationRate = 1
  } = {}) {
    super("AdaptiveLuminancePass");
    this.setFullscreenMaterial(new AdaptiveLuminanceMaterial());
    this.needsSwap = false;
    this.renderTargetPrevious = new WebGLRenderTarget2(1, 1, {
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      type: HalfFloatType,
      stencilBuffer: false,
      depthBuffer: false,
      format: RGBAFormat
    });
    this.renderTargetPrevious.texture.name = "Luminance.Previous";
    const uniforms = this.getFullscreenMaterial().uniforms;
    uniforms.luminanceBuffer0.value = this.renderTargetPrevious.texture;
    uniforms.luminanceBuffer1.value = luminanceBuffer;
    uniforms.minLuminance.value = minLuminance;
    this.renderTargetAdapted = this.renderTargetPrevious.clone();
    this.renderTargetAdapted.texture.name = "Luminance.Adapted";
    this.savePass = new SavePass(this.renderTargetPrevious, false);
    this.adaptationRate = adaptationRate;
  }
  get texture() {
    return this.renderTargetAdapted.texture;
  }
  set mipLevel1x1(value) {
    const material = this.getFullscreenMaterial();
    material.defines.MIP_LEVEL_1X1 = value.toFixed(1);
    material.needsUpdate = true;
  }
  get adaptationRate() {
    return this.getFullscreenMaterial().uniforms.tau.value;
  }
  set adaptationRate(value) {
    this.getFullscreenMaterial().uniforms.tau.value = value;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    this.getFullscreenMaterial().uniforms.deltaTime.value = deltaTime;
    renderer.setRenderTarget(this.renderToScreen ? null : this.renderTargetAdapted);
    renderer.render(this.scene, this.camera);
    this.savePass.render(renderer, this.renderTargetAdapted);
  }
};

// src/passes/BlurPass.js
import {
  LinearFilter as LinearFilter2,
  RGBFormat as RGBFormat2,
  UnsignedByteType as UnsignedByteType3,
  WebGLRenderTarget as WebGLRenderTarget3
} from "../build/three.module.js";

// src/core/Resizer.js
import {Vector2 as Vector212} from "../build/three.module.js";
var AUTO_SIZE = -1;
var Resizer = class {
  constructor(resizable, width = AUTO_SIZE, height = AUTO_SIZE, scale = 1) {
    this.resizable = resizable;
    this.base = new Vector212(1, 1);
    this.target = new Vector212(width, height);
    this.s = scale;
  }
  get scale() {
    return this.s;
  }
  set scale(value) {
    this.s = value;
    this.target.x = AUTO_SIZE;
    this.target.y = AUTO_SIZE;
    this.resizable.setSize(this.base.x, this.base.y);
  }
  get width() {
    const base = this.base;
    const target = this.target;
    let result;
    if (target.x !== AUTO_SIZE) {
      result = target.x;
    } else if (target.y !== AUTO_SIZE) {
      result = Math.round(target.y * (base.x / base.y));
    } else {
      result = Math.round(base.x * this.s);
    }
    return result;
  }
  set width(value) {
    this.target.x = value;
    this.resizable.setSize(this.base.x, this.base.y);
  }
  get height() {
    const base = this.base;
    const target = this.target;
    let result;
    if (target.y !== AUTO_SIZE) {
      result = target.y;
    } else if (target.x !== AUTO_SIZE) {
      result = Math.round(target.x / (base.x / base.y));
    } else {
      result = Math.round(base.y * this.s);
    }
    return result;
  }
  set height(value) {
    this.target.y = value;
    this.resizable.setSize(this.base.x, this.base.y);
  }
  static get AUTO_SIZE() {
    return AUTO_SIZE;
  }
};

// src/passes/BlurPass.js
var BlurPass = class extends Pass {
  constructor({
    resolutionScale = 0.5,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    kernelSize = KernelSize.LARGE
  } = {}) {
    super("BlurPass");
    this.renderTargetA = new WebGLRenderTarget3(1, 1, {
      minFilter: LinearFilter2,
      magFilter: LinearFilter2,
      stencilBuffer: false,
      depthBuffer: false
    });
    this.renderTargetA.texture.name = "Blur.Target.A";
    this.renderTargetB = this.renderTargetA.clone();
    this.renderTargetB.texture.name = "Blur.Target.B";
    this.resolution = new Resizer(this, width, height, resolutionScale);
    this.convolutionMaterial = new ConvolutionMaterial();
    this.ditheredConvolutionMaterial = new ConvolutionMaterial();
    this.ditheredConvolutionMaterial.dithering = true;
    this.dithering = false;
    this.kernelSize = kernelSize;
  }
  get width() {
    return this.resolution.width;
  }
  set width(value) {
    this.resolution.width = value;
  }
  get height() {
    return this.resolution.height;
  }
  set height(value) {
    this.resolution.height = value;
  }
  get scale() {
    return this.convolutionMaterial.uniforms.scale.value;
  }
  set scale(value) {
    this.convolutionMaterial.uniforms.scale.value = value;
    this.ditheredConvolutionMaterial.uniforms.scale.value = value;
  }
  get kernelSize() {
    return this.convolutionMaterial.kernelSize;
  }
  set kernelSize(value) {
    this.convolutionMaterial.kernelSize = value;
    this.ditheredConvolutionMaterial.kernelSize = value;
  }
  getResolutionScale() {
    return this.resolution.scale;
  }
  setResolutionScale(scale) {
    this.resolution.scale = scale;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const scene = this.scene;
    const camera = this.camera;
    const renderTargetA = this.renderTargetA;
    const renderTargetB = this.renderTargetB;
    let material = this.convolutionMaterial;
    let uniforms = material.uniforms;
    const kernel = material.getKernel();
    let lastRT = inputBuffer;
    let destRT;
    let i, l;
    this.setFullscreenMaterial(material);
    for (i = 0, l = kernel.length - 1; i < l; ++i) {
      destRT = (i & 1) === 0 ? renderTargetA : renderTargetB;
      uniforms.kernel.value = kernel[i];
      uniforms.inputBuffer.value = lastRT.texture;
      renderer.setRenderTarget(destRT);
      renderer.render(scene, camera);
      lastRT = destRT;
    }
    if (this.dithering) {
      material = this.ditheredConvolutionMaterial;
      uniforms = material.uniforms;
      this.setFullscreenMaterial(material);
    }
    uniforms.kernel.value = kernel[i];
    uniforms.inputBuffer.value = lastRT.texture;
    renderer.setRenderTarget(this.renderToScreen ? null : outputBuffer);
    renderer.render(scene, camera);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    resolution.base.set(width, height);
    const w = resolution.width;
    const h = resolution.height;
    this.renderTargetA.setSize(w, h);
    this.renderTargetB.setSize(w, h);
    this.convolutionMaterial.setTexelSize(1 / w, 1 / h);
    this.ditheredConvolutionMaterial.setTexelSize(1 / w, 1 / h);
  }
  initialize(renderer, alpha, frameBufferType) {
    if (!alpha && frameBufferType === UnsignedByteType3) {
      this.renderTargetA.texture.format = RGBFormat2;
      this.renderTargetB.texture.format = RGBFormat2;
    }
    if (frameBufferType !== void 0) {
      this.renderTargetA.texture.type = frameBufferType;
      this.renderTargetB.texture.type = frameBufferType;
      if (frameBufferType !== UnsignedByteType3) {
        const m0 = this.convolutionMaterial;
        const m1 = this.ditheredConvolutionMaterial;
        m0.defines.FRAMEBUFFER_PRECISION_HIGH = "1";
        m1.defines.FRAMEBUFFER_PRECISION_HIGH = "1";
      }
    }
  }
  static get AUTO_SIZE() {
    return Resizer.AUTO_SIZE;
  }
};

// src/passes/ClearMaskPass.js
var ClearMaskPass = class extends Pass {
  constructor() {
    super("ClearMaskPass", null, null);
    this.needsSwap = false;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const stencil = renderer.state.buffers.stencil;
    stencil.setLocked(false);
    stencil.setTest(false);
  }
};

// src/passes/ClearPass.js
import {Color} from "../build/three.module.js";
var color = new Color();
var ClearPass = class extends Pass {
  constructor(color2 = true, depth = true, stencil = false) {
    super("ClearPass", null, null);
    this.needsSwap = false;
    this.color = color2;
    this.depth = depth;
    this.stencil = stencil;
    this.overrideClearColor = null;
    this.overrideClearAlpha = -1;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const overrideClearColor = this.overrideClearColor;
    const overrideClearAlpha = this.overrideClearAlpha;
    const clearAlpha = renderer.getClearAlpha();
    const hasOverrideClearColor = overrideClearColor !== null;
    const hasOverrideClearAlpha = overrideClearAlpha >= 0;
    if (hasOverrideClearColor) {
      color.copy(renderer.getClearColor(color));
      renderer.setClearColor(overrideClearColor, hasOverrideClearAlpha ? overrideClearAlpha : clearAlpha);
    } else if (hasOverrideClearAlpha) {
      renderer.setClearAlpha(overrideClearAlpha);
    }
    renderer.setRenderTarget(this.renderToScreen ? null : inputBuffer);
    renderer.clear(this.color, this.depth, this.stencil);
    if (hasOverrideClearColor) {
      renderer.setClearColor(color, clearAlpha);
    } else if (hasOverrideClearAlpha) {
      renderer.setClearAlpha(clearAlpha);
    }
  }
};

// src/passes/DepthPass.js
import {
  Color as Color2,
  MeshDepthMaterial,
  NearestFilter as NearestFilter2,
  RGBADepthPacking,
  WebGLRenderTarget as WebGLRenderTarget4
} from "../build/three.module.js";

// src/core/OverrideMaterialManager.js
import {BackSide, DoubleSide, FrontSide} from "../build/three.module.js";
var workaroundEnabled = false;
var OverrideMaterialManager = class {
  constructor(material = null) {
    this.originalMaterials = new Map();
    this.material = null;
    this.materials = null;
    this.materialsBackSide = null;
    this.materialsDoubleSide = null;
    this.setMaterial(material);
    this.meshCount = 0;
    this.replaceMaterial = (node) => {
      if (node.isMesh) {
        let materials;
        switch (node.material.side) {
          case DoubleSide:
            materials = this.materialsDoubleSide;
            break;
          case BackSide:
            materials = this.materialsBackSide;
            break;
          default:
            materials = this.materials;
            break;
        }
        this.originalMaterials.set(node, node.material);
        if (node.isSkinnedMesh) {
          node.material = materials[2];
        } else if (node.isInstancedMesh) {
          node.material = materials[1];
        } else {
          node.material = materials[0];
        }
        ++this.meshCount;
      }
    };
  }
  setMaterial(material) {
    this.disposeMaterials();
    this.material = material;
    if (material !== null) {
      const materials = this.materials = [
        material.clone(),
        material.clone(),
        material.clone()
      ];
      for (const m2 of materials) {
        m2.uniforms = Object.assign({}, material.uniforms);
        m2.side = FrontSide;
      }
      materials[2].skinning = true;
      this.materialsBackSide = materials.map((m2) => {
        const c2 = m2.clone();
        c2.uniforms = Object.assign({}, material.uniforms);
        c2.side = BackSide;
        return c2;
      });
      this.materialsDoubleSide = materials.map((m2) => {
        const c2 = m2.clone();
        c2.uniforms = Object.assign({}, material.uniforms);
        c2.side = DoubleSide;
        return c2;
      });
    }
  }
  render(renderer, scene, camera) {
    const shadowMapEnabled = renderer.shadowMap.enabled;
    renderer.shadowMap.enabled = false;
    if (workaroundEnabled) {
      const originalMaterials = this.originalMaterials;
      this.meshCount = 0;
      scene.traverse(this.replaceMaterial);
      renderer.render(scene, camera);
      for (const entry of originalMaterials) {
        entry[0].material = entry[1];
      }
      if (this.meshCount !== originalMaterials.size) {
        originalMaterials.clear();
      }
    } else {
      const overrideMaterial = scene.overrideMaterial;
      scene.overrideMaterial = this.material;
      renderer.render(scene, camera);
      scene.overrideMaterial = overrideMaterial;
    }
    renderer.shadowMap.enabled = shadowMapEnabled;
  }
  disposeMaterials() {
    if (this.material !== null) {
      const materials = this.materials.concat(this.materialsBackSide).concat(this.materialsDoubleSide);
      for (const m2 of materials) {
        m2.dispose();
      }
    }
  }
  dispose() {
    this.originalMaterials.clear();
    this.disposeMaterials();
  }
  static get workaroundEnabled() {
    return workaroundEnabled;
  }
  static set workaroundEnabled(value) {
    workaroundEnabled = value;
  }
};

// src/passes/RenderPass.js
var RenderPass = class extends Pass {
  constructor(scene, camera, overrideMaterial = null) {
    super("RenderPass", scene, camera);
    this.needsSwap = false;
    this.clearPass = new ClearPass();
    this.overrideMaterialManager = overrideMaterial === null ? null : new OverrideMaterialManager(overrideMaterial);
  }
  get renderToScreen() {
    return super.renderToScreen;
  }
  set renderToScreen(value) {
    super.renderToScreen = value;
    this.clearPass.renderToScreen = value;
  }
  get overrideMaterial() {
    const manager = this.overrideMaterialManager;
    return manager !== null ? manager.material : null;
  }
  set overrideMaterial(value) {
    const manager = this.overrideMaterialManager;
    if (value !== null) {
      if (manager !== null) {
        manager.setMaterial(value);
      } else {
        this.overrideMaterialManager = new OverrideMaterialManager(value);
      }
    } else if (manager !== null) {
      manager.dispose();
      this.overrideMaterialManager = null;
    }
  }
  get clear() {
    return this.clearPass.enabled;
  }
  set clear(value) {
    this.clearPass.enabled = value;
  }
  getClearPass() {
    return this.clearPass;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const scene = this.scene;
    const camera = this.camera;
    const background = scene.background;
    const renderTarget = this.renderToScreen ? null : inputBuffer;
    if (this.clear) {
      if (this.clearPass.overrideClearColor !== null) {
        scene.background = null;
      }
      this.clearPass.render(renderer, inputBuffer);
    }
    renderer.setRenderTarget(renderTarget);
    if (this.overrideMaterialManager !== null) {
      this.overrideMaterialManager.render(renderer, scene, camera);
    } else {
      renderer.render(scene, camera);
    }
    if (scene.background !== background) {
      scene.background = background;
    }
  }
};

// src/passes/DepthPass.js
var DepthPass = class extends Pass {
  constructor(scene, camera, {
    resolutionScale = 1,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    renderTarget
  } = {}) {
    super("DepthPass");
    this.needsSwap = false;
    this.renderPass = new RenderPass(scene, camera, new MeshDepthMaterial({
      depthPacking: RGBADepthPacking
    }));
    const clearPass = this.renderPass.getClearPass();
    clearPass.overrideClearColor = new Color2(16777215);
    clearPass.overrideClearAlpha = 1;
    this.renderTarget = renderTarget;
    if (this.renderTarget === void 0) {
      this.renderTarget = new WebGLRenderTarget4(1, 1, {
        minFilter: NearestFilter2,
        magFilter: NearestFilter2,
        stencilBuffer: false
      });
      this.renderTarget.texture.name = "DepthPass.Target";
    }
    this.resolution = new Resizer(this, width, height, resolutionScale);
  }
  get texture() {
    return this.renderTarget.texture;
  }
  getResolutionScale() {
    return this.resolutionScale;
  }
  setResolutionScale(scale) {
    this.resolutionScale = scale;
    this.setSize(this.resolution.base.x, this.resolution.base.y);
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const renderTarget = this.renderToScreen ? null : this.renderTarget;
    this.renderPass.render(renderer, renderTarget);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    resolution.base.set(width, height);
    this.renderTarget.setSize(resolution.width, resolution.height);
  }
};

// src/passes/DepthDownsamplingPass.js
import {
  BasicDepthPacking,
  FloatType,
  NearestFilter as NearestFilter3,
  WebGLRenderTarget as WebGLRenderTarget5
} from "../build/three.module.js";
var DepthDownsamplingPass = class extends Pass {
  constructor({
    normalBuffer = null,
    resolutionScale = 0.5,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE
  } = {}) {
    super("DepthDownsamplingPass");
    this.setFullscreenMaterial(new DepthDownsamplingMaterial());
    this.needsDepthTexture = true;
    this.needsSwap = false;
    if (normalBuffer !== null) {
      const material = this.getFullscreenMaterial();
      material.uniforms.normalBuffer.value = normalBuffer;
      material.defines.DOWNSAMPLE_NORMALS = "1";
    }
    this.renderTarget = new WebGLRenderTarget5(1, 1, {
      minFilter: NearestFilter3,
      magFilter: NearestFilter3,
      stencilBuffer: false,
      depthBuffer: false,
      type: FloatType
    });
    this.renderTarget.texture.name = "DepthDownsamplingPass.Target";
    this.renderTarget.texture.generateMipmaps = false;
    this.resolution = new Resizer(this, width, height);
    this.resolution.scale = resolutionScale;
  }
  get texture() {
    return this.renderTarget.texture;
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking) {
    const material = this.getFullscreenMaterial();
    material.uniforms.depthBuffer.value = depthTexture;
    material.depthPacking = depthPacking;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    renderer.setRenderTarget(this.renderToScreen ? null : this.renderTarget);
    renderer.render(this.scene, this.camera);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    resolution.base.set(width, height);
    this.getFullscreenMaterial().setTexelSize(1 / width, 1 / height);
    this.renderTarget.setSize(resolution.width, resolution.height);
  }
  initialize(renderer, alpha, frameBufferType) {
    if (!renderer.capabilities.isWebGL2) {
      console.error("The DepthDownsamplingPass requires WebGL 2");
    }
  }
};

// src/passes/DepthPickingPass.js
import {FloatType as FloatType3, RGBADepthPacking as RGBADepthPacking3} from "../build/three.module.js";

// src/passes/DepthSavePass.js
import {
  BasicDepthPacking as BasicDepthPacking2,
  FloatType as FloatType2,
  NearestFilter as NearestFilter4,
  RGBADepthPacking as RGBADepthPacking2,
  UnsignedByteType as UnsignedByteType4,
  WebGLRenderTarget as WebGLRenderTarget6
} from "../build/three.module.js";
var DepthSavePass = class extends Pass {
  constructor({depthPacking = RGBADepthPacking2} = {}) {
    super("DepthSavePass");
    const material = new DepthCopyMaterial();
    material.setOutputDepthPacking(depthPacking);
    this.setFullscreenMaterial(material);
    this.needsDepthTexture = true;
    this.needsSwap = false;
    this.renderTarget = new WebGLRenderTarget6(1, 1, {
      type: depthPacking === RGBADepthPacking2 ? UnsignedByteType4 : FloatType2,
      minFilter: NearestFilter4,
      magFilter: NearestFilter4,
      stencilBuffer: false,
      depthBuffer: false
    });
    this.renderTarget.texture.name = "DepthSavePass.Target";
  }
  get texture() {
    return this.renderTarget.texture;
  }
  get depthPacking() {
    return this.getFullscreenMaterial().getOutputDepthPacking();
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking2) {
    const material = this.getFullscreenMaterial();
    material.uniforms.depthBuffer.value = depthTexture;
    material.setInputDepthPacking(depthPacking);
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    renderer.setRenderTarget(this.renderToScreen ? null : this.renderTarget);
    renderer.render(this.scene, this.camera);
  }
  setSize(width, height) {
    this.renderTarget.setSize(width, height);
  }
};

// src/passes/DepthPickingPass.js
var unpackFactors = new Float32Array([
  255 / 256 / 256 ** 3,
  255 / 256 / 256 ** 2,
  255 / 256 / 256
]);
function unpackRGBAToDepth(packedDepth) {
  return (packedDepth[0] * unpackFactors[0] + packedDepth[1] * unpackFactors[1] + packedDepth[2] * unpackFactors[2] + packedDepth[3]) / 256;
}
var DepthPickingPass = class extends DepthSavePass {
  constructor({
    depthPacking = RGBADepthPacking3,
    mode = DepthCopyMode.SINGLE
  } = {}) {
    super({depthPacking});
    this.name = "DepthPickingPass";
    this.getFullscreenMaterial().setMode(mode);
    this.pixelBuffer = depthPacking === RGBADepthPacking3 ? new Uint8Array(4) : new Float32Array(4);
    this.callback = null;
  }
  get mode() {
    return this.getFullscreenMaterial().getMode();
  }
  get screenPosition() {
    return this.getFullscreenMaterial().uniforms.screenPosition.value;
  }
  readDepth(ndc) {
    this.screenPosition.set(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5);
    return new Promise((resolve) => {
      this.callback = resolve;
    });
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    if (this.mode === DepthCopyMode.FULL) {
      super.render(renderer);
    }
    if (this.callback !== null) {
      const renderTarget = this.renderTarget;
      const pixelBuffer = this.pixelBuffer;
      const packed = renderTarget.texture.type !== FloatType3;
      let x = 0, y = 0;
      if (this.mode === DepthCopyMode.SINGLE) {
        super.render(renderer);
      } else {
        const screenPosition = this.screenPosition;
        x = Math.round(screenPosition.x * renderTarget.width);
        y = Math.round(screenPosition.y * renderTarget.height);
      }
      renderer.readRenderTargetPixels(renderTarget, x, y, 1, 1, pixelBuffer);
      this.callback(packed ? unpackRGBAToDepth(pixelBuffer) : pixelBuffer[0]);
      this.callback = null;
    }
  }
  setSize(width, height) {
    if (this.mode === DepthCopyMode.FULL) {
      super.setSize(width, height);
    }
  }
};

// src/passes/EffectPass.js
import {BasicDepthPacking as BasicDepthPacking3, UnsignedByteType as UnsignedByteType5} from "../build/three.module.js";

// src/effects/blending/BlendFunction.js
var BlendFunction = {
  SKIP: 0,
  ADD: 1,
  ALPHA: 2,
  AVERAGE: 3,
  COLOR_BURN: 4,
  COLOR_DODGE: 5,
  DARKEN: 6,
  DIFFERENCE: 7,
  EXCLUSION: 8,
  LIGHTEN: 9,
  MULTIPLY: 10,
  DIVIDE: 11,
  NEGATION: 12,
  NORMAL: 13,
  OVERLAY: 14,
  REFLECT: 15,
  SCREEN: 16,
  SOFT_LIGHT: 17,
  SUBTRACT: 18
};

// src/effects/blending/BlendMode.js
import {EventDispatcher, Uniform as Uniform19} from "../build/three.module.js";

// src/effects/blending/glsl/add/shader.frag
var shader_default28 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return min(x+y,1.0)*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/alpha/shader.frag
var shader_default29 = "vec3 blend(const in vec3 x,const in vec3 y,const in float opacity){return y*opacity+x*(1.0-opacity);}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){float a=min(y.a,opacity);return vec4(blend(x.rgb,y.rgb,a),max(x.a,a));}";

// src/effects/blending/glsl/average/shader.frag
var shader_default30 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return(x+y)*0.5*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/color-burn/shader.frag
var shader_default31 = "float blend(const in float x,const in float y){return(y==0.0)? y : max(1.0-(1.0-x)/y,0.0);}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){vec4 z=vec4(blend(x.r,y.r),blend(x.g,y.g),blend(x.b,y.b),blend(x.a,y.a));return z*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/color-dodge/shader.frag
var shader_default32 = "float blend(const in float x,const in float y){return(y==1.0)? y : min(x/(1.0-y),1.0);}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){vec4 z=vec4(blend(x.r,y.r),blend(x.g,y.g),blend(x.b,y.b),blend(x.a,y.a));return z*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/darken/shader.frag
var shader_default33 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return min(x,y)*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/difference/shader.frag
var shader_default34 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return abs(x-y)*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/exclusion/shader.frag
var shader_default35 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return(x+y-2.0*x*y)*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/lighten/shader.frag
var shader_default36 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return max(x,y)*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/multiply/shader.frag
var shader_default37 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return x*y*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/divide/shader.frag
var shader_default38 = "float blend(const in float x,const in float y){return(y>0.0)? min(x/y,1.0): 1.0;}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){vec4 z=vec4(blend(x.r,y.r),blend(x.g,y.g),blend(x.b,y.b),blend(x.a,y.a));return z*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/negation/shader.frag
var shader_default39 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return(1.0-abs(1.0-x-y))*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/normal/shader.frag
var shader_default40 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return y*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/overlay/shader.frag
var shader_default41 = "float blend(const in float x,const in float y){return(x<0.5)?(2.0*x*y):(1.0-2.0*(1.0-x)*(1.0-y));}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){vec4 z=vec4(blend(x.r,y.r),blend(x.g,y.g),blend(x.b,y.b),blend(x.a,y.a));return z*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/reflect/shader.frag
var shader_default42 = "float blend(const in float x,const in float y){return(y==1.0)? y : min(x*x/(1.0-y),1.0);}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){vec4 z=vec4(blend(x.r,y.r),blend(x.g,y.g),blend(x.b,y.b),blend(x.a,y.a));return z*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/screen/shader.frag
var shader_default43 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return(1.0-(1.0-x)*(1.0-y))*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/soft-light/shader.frag
var shader_default44 = "float blend(const in float x,const in float y){return(y<0.5)?(2.0*x*y+x*x*(1.0-2.0*y)):(sqrt(x)*(2.0*y-1.0)+2.0*x*(1.0-y));}vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){vec4 z=vec4(blend(x.r,y.r),blend(x.g,y.g),blend(x.b,y.b),blend(x.a,y.a));return z*opacity+x*(1.0-opacity);}";

// src/effects/blending/glsl/subtract/shader.frag
var shader_default45 = "vec4 blend(const in vec4 x,const in vec4 y,const in float opacity){return max(x+y-1.0,0.0)*opacity+x*(1.0-opacity);}";

// src/effects/blending/BlendMode.js
var blendFunctions = new Map([
  [BlendFunction.SKIP, null],
  [BlendFunction.ADD, shader_default28],
  [BlendFunction.ALPHA, shader_default29],
  [BlendFunction.AVERAGE, shader_default30],
  [BlendFunction.COLOR_BURN, shader_default31],
  [BlendFunction.COLOR_DODGE, shader_default32],
  [BlendFunction.DARKEN, shader_default33],
  [BlendFunction.DIFFERENCE, shader_default34],
  [BlendFunction.EXCLUSION, shader_default35],
  [BlendFunction.LIGHTEN, shader_default36],
  [BlendFunction.MULTIPLY, shader_default37],
  [BlendFunction.DIVIDE, shader_default38],
  [BlendFunction.NEGATION, shader_default39],
  [BlendFunction.NORMAL, shader_default40],
  [BlendFunction.OVERLAY, shader_default41],
  [BlendFunction.REFLECT, shader_default42],
  [BlendFunction.SCREEN, shader_default43],
  [BlendFunction.SOFT_LIGHT, shader_default44],
  [BlendFunction.SUBTRACT, shader_default45]
]);
var BlendMode = class extends EventDispatcher {
  constructor(blendFunction, opacity = 1) {
    super();
    this.blendFunction = blendFunction;
    this.opacity = new Uniform19(opacity);
  }
  getBlendFunction() {
    return this.blendFunction;
  }
  setBlendFunction(blendFunction) {
    this.blendFunction = blendFunction;
    this.dispatchEvent({type: "change"});
  }
  getShaderCode() {
    return blendFunctions.get(this.blendFunction);
  }
};

// src/effects/Effect.js
import {EventDispatcher as EventDispatcher2, Scene as Scene2} from "../build/three.module.js";
var Effect = class extends EventDispatcher2 {
  constructor(name, fragmentShader, {
    attributes = EffectAttribute.NONE,
    blendFunction = BlendFunction.SCREEN,
    defines = new Map(),
    uniforms = new Map(),
    extensions = null,
    vertexShader = null
  } = {}) {
    super();
    this.name = name;
    this.attributes = attributes;
    this.fragmentShader = fragmentShader;
    this.vertexShader = vertexShader;
    this.defines = defines;
    this.uniforms = uniforms;
    this.extensions = extensions;
    this.blendMode = new BlendMode(blendFunction);
    this.blendMode.addEventListener("change", (event) => this.setChanged());
  }
  setChanged() {
    this.dispatchEvent({type: "change"});
  }
  getAttributes() {
    return this.attributes;
  }
  setAttributes(attributes) {
    this.attributes = attributes;
    this.setChanged();
  }
  getFragmentShader() {
    return this.fragmentShader;
  }
  setFragmentShader(fragmentShader) {
    this.fragmentShader = fragmentShader;
    this.setChanged();
  }
  getVertexShader() {
    return this.vertexShader;
  }
  setVertexShader(vertexShader) {
    this.vertexShader = vertexShader;
    this.setChanged();
  }
  setDepthTexture(depthTexture, depthPacking = 0) {
  }
  update(renderer, inputBuffer, deltaTime) {
  }
  setSize(width, height) {
  }
  initialize(renderer, alpha, frameBufferType) {
  }
  dispose() {
    for (const key of Object.keys(this)) {
      const property = this[key];
      if (property !== null && typeof property.dispose === "function") {
        if (property instanceof Scene2) {
          continue;
        }
        this[key].dispose();
      }
    }
  }
};
var EffectAttribute = {
  NONE: 0,
  DEPTH: 1,
  CONVOLUTION: 2
};
var WebGLExtension = {
  DERIVATIVES: "derivatives",
  FRAG_DEPTH: "fragDepth",
  DRAW_BUFFERS: "drawBuffers",
  SHADER_TEXTURE_LOD: "shaderTextureLOD"
};

// src/passes/EffectPass.js
function findSubstrings(regExp, string) {
  const substrings = [];
  let result;
  while ((result = regExp.exec(string)) !== null) {
    substrings.push(result[1]);
  }
  return substrings;
}
function prefixSubstrings(prefix, substrings, strings) {
  let prefixed, regExp;
  for (const substring of substrings) {
    prefixed = "$1" + prefix + substring.charAt(0).toUpperCase() + substring.slice(1);
    regExp = new RegExp("([^\\.])(\\b" + substring + "\\b)", "g");
    for (const entry of strings.entries()) {
      if (entry[1] !== null) {
        strings.set(entry[0], entry[1].replace(regExp, prefixed));
      }
    }
  }
}
function integrateEffect(prefix, effect, shaderParts, blendModes, defines, uniforms, attributes) {
  const functionRegExp = /(?:\w+\s+(\w+)\([\w\s,]*\)\s*{[^}]+})/g;
  const varyingRegExp = /(?:varying\s+\w+\s+(\w*))/g;
  const blendMode = effect.blendMode;
  const shaders = new Map([
    ["fragment", effect.getFragmentShader()],
    ["vertex", effect.getVertexShader()]
  ]);
  const mainImageExists = shaders.get("fragment") !== void 0 && /mainImage/.test(shaders.get("fragment"));
  const mainUvExists = shaders.get("fragment") !== void 0 && /mainUv/.test(shaders.get("fragment"));
  let varyings = [], names = [];
  let transformedUv = false;
  let readDepth = false;
  if (shaders.get("fragment") === void 0) {
    console.error("Missing fragment shader", effect);
  } else if (mainUvExists && (attributes & EffectAttribute.CONVOLUTION) !== 0) {
    console.error("Effects that transform UV coordinates are incompatible with convolution effects", effect);
  } else if (!mainImageExists && !mainUvExists) {
    console.error("The fragment shader contains neither a mainImage nor a mainUv function", effect);
  } else {
    if (mainUvExists) {
      shaderParts.set(Section.FRAGMENT_MAIN_UV, shaderParts.get(Section.FRAGMENT_MAIN_UV) + "	" + prefix + "MainUv(UV);\n");
      transformedUv = true;
    }
    if (shaders.get("vertex") !== null && /mainSupport/.test(shaders.get("vertex"))) {
      let string = "	" + prefix + "MainSupport(";
      if (/mainSupport *\([\w\s]*?uv\s*?\)/.test(shaders.get("vertex"))) {
        string += "vUv";
      }
      string += ");\n";
      shaderParts.set(Section.VERTEX_MAIN_SUPPORT, shaderParts.get(Section.VERTEX_MAIN_SUPPORT) + string);
      varyings = varyings.concat(findSubstrings(varyingRegExp, shaders.get("vertex")));
      names = names.concat(varyings).concat(findSubstrings(functionRegExp, shaders.get("vertex")));
    }
    names = names.concat(findSubstrings(functionRegExp, shaders.get("fragment"))).concat(Array.from(effect.defines.keys()).map((s) => s.replace(/\([\w\s,]*\)/g, ""))).concat(Array.from(effect.uniforms.keys()));
    effect.uniforms.forEach((value, key) => uniforms.set(prefix + key.charAt(0).toUpperCase() + key.slice(1), value));
    effect.defines.forEach((value, key) => defines.set(prefix + key.charAt(0).toUpperCase() + key.slice(1), value));
    prefixSubstrings(prefix, names, defines);
    prefixSubstrings(prefix, names, shaders);
    blendModes.set(blendMode.blendFunction, blendMode);
    if (mainImageExists) {
      const depthParamRegExp = /MainImage *\([\w\s,]*?depth[\w\s,]*?\)/;
      let string = prefix + "MainImage(color0, UV, ";
      if ((attributes & EffectAttribute.DEPTH) !== 0 && depthParamRegExp.test(shaders.get("fragment"))) {
        string += "depth, ";
        readDepth = true;
      }
      string += "color1);\n	";
      const blendOpacity = prefix + "BlendOpacity";
      uniforms.set(blendOpacity, blendMode.opacity);
      string += "color0 = blend" + blendMode.getBlendFunction() + "(color0, color1, " + blendOpacity + ");\n\n	";
      shaderParts.set(Section.FRAGMENT_MAIN_IMAGE, shaderParts.get(Section.FRAGMENT_MAIN_IMAGE) + string);
      shaderParts.set(Section.FRAGMENT_HEAD, shaderParts.get(Section.FRAGMENT_HEAD) + "uniform float " + blendOpacity + ";\n\n");
    }
    shaderParts.set(Section.FRAGMENT_HEAD, shaderParts.get(Section.FRAGMENT_HEAD) + shaders.get("fragment") + "\n");
    if (shaders.get("vertex") !== null) {
      shaderParts.set(Section.VERTEX_HEAD, shaderParts.get(Section.VERTEX_HEAD) + shaders.get("vertex") + "\n");
    }
  }
  return {varyings, transformedUv, readDepth};
}
var EffectPass = class extends Pass {
  constructor(camera, ...effects) {
    super("EffectPass");
    this.setFullscreenMaterial(new EffectMaterial(null, null, null, camera));
    this.effects = effects.sort((a, b) => b.attributes - a.attributes);
    this.skipRendering = false;
    this.uniforms = 0;
    this.varyings = 0;
    this.minTime = 1;
    this.maxTime = Number.POSITIVE_INFINITY;
  }
  get encodeOutput() {
    return this.getFullscreenMaterial().defines.ENCODE_OUTPUT !== void 0;
  }
  set encodeOutput(value) {
    if (this.encodeOutput !== value) {
      const material = this.getFullscreenMaterial();
      material.needsUpdate = true;
      if (value) {
        material.defines.ENCODE_OUTPUT = "1";
      } else {
        delete material.defines.ENCODE_OUTPUT;
      }
    }
  }
  get dithering() {
    return this.getFullscreenMaterial().dithering;
  }
  set dithering(value) {
    const material = this.getFullscreenMaterial();
    if (material.dithering !== value) {
      material.dithering = value;
      material.needsUpdate = true;
    }
  }
  verifyResources(renderer) {
    const capabilities = renderer.capabilities;
    let max = Math.min(capabilities.maxFragmentUniforms, capabilities.maxVertexUniforms);
    if (this.uniforms > max) {
      console.warn("The current rendering context doesn't support more than " + max + " uniforms, but " + this.uniforms + " were defined");
    }
    max = capabilities.maxVaryings;
    if (this.varyings > max) {
      console.warn("The current rendering context doesn't support more than " + max + " varyings, but " + this.varyings + " were defined");
    }
  }
  updateMaterial() {
    const blendRegExp = /\bblend\b/g;
    const shaderParts = new Map([
      [Section.FRAGMENT_HEAD, ""],
      [Section.FRAGMENT_MAIN_UV, ""],
      [Section.FRAGMENT_MAIN_IMAGE, ""],
      [Section.VERTEX_HEAD, ""],
      [Section.VERTEX_MAIN_SUPPORT, ""]
    ]);
    const blendModes = new Map();
    const defines = new Map();
    const uniforms = new Map();
    const extensions = new Set();
    let id = 0, varyings = 0, attributes = 0;
    let transformedUv = false;
    let readDepth = false;
    let result;
    for (const effect of this.effects) {
      if (effect.blendMode.getBlendFunction() === BlendFunction.SKIP) {
        attributes |= effect.getAttributes() & EffectAttribute.DEPTH;
      } else if ((attributes & EffectAttribute.CONVOLUTION) !== 0 && (effect.getAttributes() & EffectAttribute.CONVOLUTION) !== 0) {
        console.error("Convolution effects cannot be merged", effect);
      } else {
        attributes |= effect.getAttributes();
        result = integrateEffect("e" + id++, effect, shaderParts, blendModes, defines, uniforms, attributes);
        varyings += result.varyings.length;
        transformedUv = transformedUv || result.transformedUv;
        readDepth = readDepth || result.readDepth;
        if (effect.extensions !== null) {
          for (const extension of effect.extensions) {
            extensions.add(extension);
          }
        }
      }
    }
    for (const blendMode of blendModes.values()) {
      shaderParts.set(Section.FRAGMENT_HEAD, shaderParts.get(Section.FRAGMENT_HEAD) + blendMode.getShaderCode().replace(blendRegExp, "blend" + blendMode.getBlendFunction()) + "\n");
    }
    if ((attributes & EffectAttribute.DEPTH) !== 0) {
      if (readDepth) {
        shaderParts.set(Section.FRAGMENT_MAIN_IMAGE, "float depth = readDepth(UV);\n\n	" + shaderParts.get(Section.FRAGMENT_MAIN_IMAGE));
      }
      this.needsDepthTexture = this.getDepthTexture() === null;
    } else {
      this.needsDepthTexture = false;
    }
    if (transformedUv) {
      shaderParts.set(Section.FRAGMENT_MAIN_UV, "vec2 transformedUv = vUv;\n" + shaderParts.get(Section.FRAGMENT_MAIN_UV));
      defines.set("UV", "transformedUv");
    } else {
      defines.set("UV", "vUv");
    }
    shaderParts.forEach((value, key, map) => map.set(key, value.trim().replace(/^#/, "\n#")));
    this.uniforms = uniforms.size;
    this.varyings = varyings;
    this.skipRendering = id === 0;
    this.needsSwap = !this.skipRendering;
    const material = this.getFullscreenMaterial();
    material.setShaderParts(shaderParts).setDefines(defines).setUniforms(uniforms);
    material.extensions = {};
    if (extensions.size > 0) {
      for (const extension of extensions) {
        material.extensions[extension] = true;
      }
    }
    this.needsUpdate = false;
  }
  recompile(renderer) {
    this.updateMaterial();
    if (renderer !== void 0) {
      this.verifyResources(renderer);
    }
  }
  getDepthTexture() {
    return this.getFullscreenMaterial().uniforms.depthBuffer.value;
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking3) {
    const material = this.getFullscreenMaterial();
    material.uniforms.depthBuffer.value = depthTexture;
    material.depthPacking = depthPacking;
    material.needsUpdate = true;
    for (const effect of this.effects) {
      effect.setDepthTexture(depthTexture, depthPacking);
    }
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const material = this.getFullscreenMaterial();
    const time = material.uniforms.time.value + deltaTime;
    if (this.needsUpdate) {
      this.recompile(renderer);
    }
    for (const effect of this.effects) {
      effect.update(renderer, inputBuffer, deltaTime);
    }
    if (!this.skipRendering || this.renderToScreen) {
      material.uniforms.inputBuffer.value = inputBuffer.texture;
      material.uniforms.time.value = time <= this.maxTime ? time : this.minTime;
      renderer.setRenderTarget(this.renderToScreen ? null : outputBuffer);
      renderer.render(this.scene, this.camera);
    }
  }
  setSize(width, height) {
    this.getFullscreenMaterial().setSize(width, height);
    for (const effect of this.effects) {
      effect.setSize(width, height);
    }
  }
  initialize(renderer, alpha, frameBufferType) {
    for (const effect of this.effects) {
      effect.initialize(renderer, alpha, frameBufferType);
      effect.addEventListener("change", (event) => this.handleEvent(event));
    }
    this.updateMaterial();
    this.verifyResources(renderer);
    if (frameBufferType !== void 0 && frameBufferType !== UnsignedByteType5) {
      const material = this.getFullscreenMaterial();
      material.defines.FRAMEBUFFER_PRECISION_HIGH = "1";
    }
  }
  dispose() {
    super.dispose();
    for (const effect of this.effects) {
      effect.dispose();
    }
  }
  handleEvent(event) {
    switch (event.type) {
      case "change":
        this.needsUpdate = true;
        break;
    }
  }
};

// src/passes/LambdaPass.js
var LambdaPass = class extends Pass {
  constructor(f) {
    super("LambdaPass", null, null);
    this.needsSwap = false;
    this.f = f;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    this.f();
  }
};

// src/passes/LuminancePass.js
import {
  LinearFilter as LinearFilter3,
  LuminanceFormat,
  RGBAFormat as RGBAFormat2,
  UnsignedByteType as UnsignedByteType6,
  WebGLRenderTarget as WebGLRenderTarget7
} from "../build/three.module.js";
var LuminancePass = class extends Pass {
  constructor({
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    renderTarget,
    luminanceRange,
    colorOutput
  } = {}) {
    super("LuminancePass");
    this.setFullscreenMaterial(new LuminanceMaterial(colorOutput, luminanceRange));
    this.needsSwap = false;
    this.renderTarget = renderTarget;
    if (this.renderTarget === void 0) {
      this.renderTarget = new WebGLRenderTarget7(1, 1, {
        minFilter: LinearFilter3,
        magFilter: LinearFilter3,
        format: colorOutput ? RGBAFormat2 : LuminanceFormat,
        stencilBuffer: false,
        depthBuffer: false
      });
      this.renderTarget.texture.name = "LuminancePass.Target";
      this.renderTarget.texture.generateMipmaps = false;
    }
    this.resolution = new Resizer(this, width, height);
  }
  get texture() {
    return this.renderTarget.texture;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const material = this.getFullscreenMaterial();
    material.uniforms.inputBuffer.value = inputBuffer.texture;
    renderer.setRenderTarget(this.renderToScreen ? null : this.renderTarget);
    renderer.render(this.scene, this.camera);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    resolution.base.set(width, height);
    this.renderTarget.setSize(resolution.width, resolution.height);
  }
  initialize(renderer, alpha, frameBufferType) {
    if (frameBufferType !== void 0 && frameBufferType !== UnsignedByteType6) {
      const material = this.getFullscreenMaterial();
      material.defines.FRAMEBUFFER_PRECISION_HIGH = "1";
    }
  }
};

// src/passes/MaskPass.js
var MaskPass = class extends Pass {
  constructor(scene, camera) {
    super("MaskPass", scene, camera);
    this.needsSwap = false;
    this.clearPass = new ClearPass(false, false, true);
    this.inverse = false;
  }
  get clear() {
    return this.clearPass.enabled;
  }
  set clear(value) {
    this.clearPass.enabled = value;
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const context = renderer.getContext();
    const buffers = renderer.state.buffers;
    const scene = this.scene;
    const camera = this.camera;
    const clearPass = this.clearPass;
    const writeValue = this.inverse ? 0 : 1;
    const clearValue = 1 - writeValue;
    buffers.color.setMask(false);
    buffers.depth.setMask(false);
    buffers.color.setLocked(true);
    buffers.depth.setLocked(true);
    buffers.stencil.setTest(true);
    buffers.stencil.setOp(context.REPLACE, context.REPLACE, context.REPLACE);
    buffers.stencil.setFunc(context.ALWAYS, writeValue, 4294967295);
    buffers.stencil.setClear(clearValue);
    buffers.stencil.setLocked(true);
    if (this.clear) {
      if (this.renderToScreen) {
        clearPass.render(renderer, null);
      } else {
        clearPass.render(renderer, inputBuffer);
        clearPass.render(renderer, outputBuffer);
      }
    }
    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
    } else {
      renderer.setRenderTarget(inputBuffer);
      renderer.render(scene, camera);
      renderer.setRenderTarget(outputBuffer);
      renderer.render(scene, camera);
    }
    buffers.color.setLocked(false);
    buffers.depth.setLocked(false);
    buffers.stencil.setLocked(false);
    buffers.stencil.setFunc(context.EQUAL, 1, 4294967295);
    buffers.stencil.setOp(context.KEEP, context.KEEP, context.KEEP);
    buffers.stencil.setLocked(true);
  }
};

// src/passes/NormalPass.js
import {
  Color as Color3,
  MeshNormalMaterial,
  NearestFilter as NearestFilter5,
  RGBFormat as RGBFormat3,
  WebGLRenderTarget as WebGLRenderTarget8
} from "../build/three.module.js";
var NormalPass = class extends Pass {
  constructor(scene, camera, {
    resolutionScale = 1,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    renderTarget
  } = {}) {
    super("NormalPass");
    this.needsSwap = false;
    this.renderPass = new RenderPass(scene, camera, new MeshNormalMaterial());
    const clearPass = this.renderPass.getClearPass();
    clearPass.overrideClearColor = new Color3(7829503);
    clearPass.overrideClearAlpha = 1;
    this.renderTarget = renderTarget;
    if (this.renderTarget === void 0) {
      this.renderTarget = new WebGLRenderTarget8(1, 1, {
        minFilter: NearestFilter5,
        magFilter: NearestFilter5,
        format: RGBFormat3,
        stencilBuffer: false
      });
      this.renderTarget.texture.name = "NormalPass.Target";
    }
    this.resolution = new Resizer(this, width, height, resolutionScale);
  }
  get texture() {
    return this.renderTarget.texture;
  }
  getResolutionScale() {
    return this.resolutionScale;
  }
  setResolutionScale(scale) {
    this.resolutionScale = scale;
    this.setSize(this.resolution.base.x, this.resolution.base.y);
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    const renderTarget = this.renderToScreen ? null : this.renderTarget;
    this.renderPass.render(renderer, renderTarget, renderTarget);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    resolution.base.set(width, height);
    this.renderTarget.setSize(resolution.width, resolution.height);
  }
};

// src/passes/ShaderPass.js
import {UnsignedByteType as UnsignedByteType7} from "../build/three.module.js";
var ShaderPass = class extends Pass {
  constructor(material, input = "inputBuffer") {
    super("ShaderPass");
    this.setFullscreenMaterial(material);
    this.uniform = null;
    this.setInput(input);
  }
  setInput(input) {
    const material = this.getFullscreenMaterial();
    this.uniform = null;
    if (material !== null) {
      const uniforms = material.uniforms;
      if (uniforms !== void 0 && uniforms[input] !== void 0) {
        this.uniform = uniforms[input];
      }
    }
  }
  render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest) {
    if (this.uniform !== null && inputBuffer !== null) {
      this.uniform.value = inputBuffer.texture;
    }
    renderer.setRenderTarget(this.renderToScreen ? null : outputBuffer);
    renderer.render(this.scene, this.camera);
  }
  initialize(renderer, alpha, frameBufferType) {
    if (frameBufferType !== void 0 && frameBufferType !== UnsignedByteType7) {
      const material = this.getFullscreenMaterial();
      material.defines.FRAMEBUFFER_PRECISION_HIGH = "1";
    }
  }
};

// src/core/EffectComposer.js
var EffectComposer = class {
  constructor(renderer = null, {
    depthBuffer = true,
    stencilBuffer = false,
    multisampling = 0,
    frameBufferType
  } = {}) {
    this.renderer = renderer;
    this.inputBuffer = null;
    this.outputBuffer = null;
    if (this.renderer !== null) {
      this.renderer.autoClear = false;
      this.inputBuffer = this.createBuffer(depthBuffer, stencilBuffer, frameBufferType, multisampling);
      this.outputBuffer = this.inputBuffer.clone();
    }
    this.copyPass = new ShaderPass(new CopyMaterial());
    this.depthTexture = null;
    this.passes = [];
    this.autoRenderToScreen = true;
  }
  get multisampling() {
    return this.inputBuffer instanceof WebGLMultisampleRenderTarget ? this.inputBuffer.samples : 0;
  }
  set multisampling(value) {
    const buffer = this.inputBuffer;
    const multisampling = this.multisampling;
    if (multisampling > 0 && value > 0) {
      this.inputBuffer.samples = value;
      this.outputBuffer.samples = value;
    } else if (multisampling !== value) {
      this.inputBuffer.dispose();
      this.outputBuffer.dispose();
      this.inputBuffer = this.createBuffer(buffer.depthBuffer, buffer.stencilBuffer, buffer.texture.type, value);
      this.inputBuffer.depthTexture = this.depthTexture;
      this.outputBuffer = this.inputBuffer.clone();
    }
  }
  getRenderer() {
    return this.renderer;
  }
  replaceRenderer(renderer, updateDOM = true) {
    const oldRenderer = this.renderer;
    if (oldRenderer !== null && oldRenderer !== renderer) {
      const oldSize = oldRenderer.getSize(new Vector213());
      const newSize = renderer.getSize(new Vector213());
      const parent = oldRenderer.domElement.parentNode;
      this.renderer = renderer;
      this.renderer.autoClear = false;
      if (!oldSize.equals(newSize)) {
        this.setSize();
      }
      if (updateDOM && parent !== null) {
        parent.removeChild(oldRenderer.domElement);
        parent.appendChild(renderer.domElement);
      }
    }
    return oldRenderer;
  }
  createDepthTexture() {
    const depthTexture = this.depthTexture = new DepthTexture();
    this.inputBuffer.depthTexture = depthTexture;
    this.inputBuffer.dispose();
    if (this.inputBuffer.stencilBuffer) {
      depthTexture.format = DepthStencilFormat;
      depthTexture.type = UnsignedInt248Type;
    } else {
      depthTexture.type = UnsignedIntType;
    }
    return depthTexture;
  }
  deleteDepthTexture() {
    if (this.depthTexture !== null) {
      this.depthTexture.dispose();
      this.depthTexture = null;
      this.inputBuffer.depthTexture = null;
      this.inputBuffer.dispose();
      for (const pass of this.passes) {
        pass.setDepthTexture(null);
      }
    }
  }
  createBuffer(depthBuffer, stencilBuffer, type, multisampling) {
    const size = this.renderer.getDrawingBufferSize(new Vector213());
    const alpha = this.renderer.getContext().getContextAttributes().alpha;
    const options = {
      format: !alpha && type === UnsignedByteType8 ? RGBFormat4 : RGBAFormat3,
      minFilter: LinearFilter4,
      magFilter: LinearFilter4,
      stencilBuffer,
      depthBuffer,
      type
    };
    const renderTarget = multisampling > 0 ? new WebGLMultisampleRenderTarget(size.width, size.height, options) : new WebGLRenderTarget9(size.width, size.height, options);
    if (multisampling > 0) {
      renderTarget.samples = multisampling;
    }
    renderTarget.texture.name = "EffectComposer.Buffer";
    renderTarget.texture.generateMipmaps = false;
    return renderTarget;
  }
  addPass(pass, index) {
    const passes = this.passes;
    const renderer = this.renderer;
    const drawingBufferSize = renderer.getDrawingBufferSize(new Vector213());
    const alpha = renderer.getContext().getContextAttributes().alpha;
    const frameBufferType = this.inputBuffer.texture.type;
    pass.setSize(drawingBufferSize.width, drawingBufferSize.height);
    pass.initialize(renderer, alpha, frameBufferType);
    if (this.autoRenderToScreen) {
      if (passes.length > 0) {
        passes[passes.length - 1].renderToScreen = false;
      }
      if (pass.renderToScreen) {
        this.autoRenderToScreen = false;
      }
    }
    if (index !== void 0) {
      passes.splice(index, 0, pass);
    } else {
      passes.push(pass);
    }
    if (this.autoRenderToScreen) {
      passes[passes.length - 1].renderToScreen = true;
    }
    if (pass.needsDepthTexture || this.depthTexture !== null) {
      if (this.depthTexture === null) {
        const depthTexture = this.createDepthTexture();
        for (pass of passes) {
          pass.setDepthTexture(depthTexture);
        }
      } else {
        pass.setDepthTexture(this.depthTexture);
      }
    }
  }
  removePass(pass) {
    const passes = this.passes;
    const index = passes.indexOf(pass);
    const exists = index !== -1;
    const removed = exists && passes.splice(index, 1).length > 0;
    if (removed) {
      if (this.depthTexture !== null) {
        const reducer = (a, b) => a || b.needsDepthTexture;
        const depthTextureRequired = passes.reduce(reducer, false);
        if (!depthTextureRequired) {
          if (pass.getDepthTexture() === this.depthTexture) {
            pass.setDepthTexture(null);
          }
          this.deleteDepthTexture();
        }
      }
      if (this.autoRenderToScreen) {
        if (index === passes.length) {
          pass.renderToScreen = false;
          if (passes.length > 0) {
            passes[passes.length - 1].renderToScreen = true;
          }
        }
      }
    }
  }
  removeAllPasses() {
    const passes = this.passes;
    this.deleteDepthTexture();
    if (passes.length > 0) {
      if (this.autoRenderToScreen) {
        passes[passes.length - 1].renderToScreen = false;
      }
      this.passes = [];
    }
  }
  render(deltaTime) {
    const renderer = this.renderer;
    const copyPass = this.copyPass;
    let inputBuffer = this.inputBuffer;
    let outputBuffer = this.outputBuffer;
    let stencilTest = false;
    let context, stencil, buffer;
    for (const pass of this.passes) {
      if (pass.enabled) {
        pass.render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest);
        if (pass.needsSwap) {
          if (stencilTest) {
            copyPass.renderToScreen = pass.renderToScreen;
            context = renderer.getContext();
            stencil = renderer.state.buffers.stencil;
            stencil.setFunc(context.NOTEQUAL, 1, 4294967295);
            copyPass.render(renderer, inputBuffer, outputBuffer, deltaTime, stencilTest);
            stencil.setFunc(context.EQUAL, 1, 4294967295);
          }
          buffer = inputBuffer;
          inputBuffer = outputBuffer;
          outputBuffer = buffer;
        }
        if (pass instanceof MaskPass) {
          stencilTest = true;
        } else if (pass instanceof ClearMaskPass) {
          stencilTest = false;
        }
      }
    }
  }
  setSize(width, height, updateStyle) {
    const renderer = this.renderer;
    if (width === void 0 || height === void 0) {
      const size = renderer.getSize(new Vector213());
      width = size.width;
      height = size.height;
    } else {
      renderer.setSize(width, height, updateStyle);
    }
    const drawingBufferSize = renderer.getDrawingBufferSize(new Vector213());
    this.inputBuffer.setSize(drawingBufferSize.width, drawingBufferSize.height);
    this.outputBuffer.setSize(drawingBufferSize.width, drawingBufferSize.height);
    for (const pass of this.passes) {
      pass.setSize(drawingBufferSize.width, drawingBufferSize.height);
    }
  }
  reset() {
    this.dispose();
    this.autoRenderToScreen = true;
  }
  dispose() {
    for (const pass of this.passes) {
      pass.dispose();
    }
    this.passes = [];
    if (this.inputBuffer !== null) {
      this.inputBuffer.dispose();
    }
    if (this.outputBuffer !== null) {
      this.outputBuffer.dispose();
    }
    this.deleteDepthTexture();
    this.copyPass.dispose();
  }
};

// src/core/Initializable.js
var Initializable = class {
  initialize(renderer, alpha, frameBufferType) {
  }
};

// src/core/Resizable.js
var Resizable = class {
  setSize(width, height) {
  }
};

// src/core/Selection.js
var Selection = class extends Set {
  constructor(iterable, layer = 10) {
    super();
    this.currentLayer = layer;
    if (iterable !== void 0) {
      this.set(iterable);
    }
  }
  get layer() {
    return this.currentLayer;
  }
  set layer(value) {
    const currentLayer = this.currentLayer;
    for (const object of this) {
      object.layers.disable(currentLayer);
      object.layers.enable(value);
    }
    this.currentLayer = value;
  }
  clear() {
    const layer = this.layer;
    for (const object of this) {
      object.layers.disable(layer);
    }
    return super.clear();
  }
  set(objects) {
    this.clear();
    for (const object of objects) {
      this.add(object);
    }
    return this;
  }
  indexOf(object) {
    return this.has(object) ? 0 : -1;
  }
  add(object) {
    object.layers.enable(this.layer);
    super.add(object);
    return this;
  }
  delete(object) {
    if (this.has(object)) {
      object.layers.disable(this.layer);
    }
    return super.delete(object);
  }
  setVisible(visible) {
    for (const object of this) {
      if (visible) {
        object.layers.enable(0);
      } else {
        object.layers.disable(0);
      }
    }
    return this;
  }
};

// src/effects/BloomEffect.js
import {
  LinearFilter as LinearFilter5,
  RGBFormat as RGBFormat5,
  Uniform as Uniform20,
  UnsignedByteType as UnsignedByteType9,
  WebGLRenderTarget as WebGLRenderTarget10
} from "../build/three.module.js";

// src/effects/glsl/bloom/shader.frag
var shader_default46 = "#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D map;\n#else\nuniform lowp sampler2D map;\n#endif\nuniform float intensity;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){outputColor=clamp(texture2D(map,uv)*intensity,0.0,1.0);}";

// src/effects/BloomEffect.js
var BloomEffect = class extends Effect {
  constructor({
    blendFunction = BlendFunction.SCREEN,
    luminanceThreshold = 0.9,
    luminanceSmoothing = 0.025,
    resolutionScale = 0.5,
    intensity = 1,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    kernelSize = KernelSize.LARGE
  } = {}) {
    super("BloomEffect", shader_default46, {
      blendFunction,
      uniforms: new Map([
        ["map", new Uniform20(null)],
        ["intensity", new Uniform20(intensity)]
      ])
    });
    this.renderTarget = new WebGLRenderTarget10(1, 1, {
      minFilter: LinearFilter5,
      magFilter: LinearFilter5,
      stencilBuffer: false,
      depthBuffer: false
    });
    this.renderTarget.texture.name = "Bloom.Target";
    this.renderTarget.texture.generateMipmaps = false;
    this.uniforms.get("map").value = this.renderTarget.texture;
    this.blurPass = new BlurPass({resolutionScale, width, height, kernelSize});
    this.blurPass.resolution.resizable = this;
    this.luminancePass = new LuminancePass({
      renderTarget: this.renderTarget,
      colorOutput: true
    });
    this.luminancePass.resolution = this.resolution;
    this.luminanceMaterial.threshold = luminanceThreshold;
    this.luminanceMaterial.smoothing = luminanceSmoothing;
  }
  get texture() {
    return this.renderTarget.texture;
  }
  get luminanceMaterial() {
    return this.luminancePass.getFullscreenMaterial();
  }
  get resolution() {
    return this.blurPass.resolution;
  }
  get width() {
    return this.resolution.width;
  }
  set width(value) {
    this.resolution.width = value;
  }
  get height() {
    return this.resolution.height;
  }
  set height(value) {
    this.resolution.height = value;
  }
  get dithering() {
    return this.blurPass.dithering;
  }
  set dithering(value) {
    this.blurPass.dithering = value;
  }
  get kernelSize() {
    return this.blurPass.kernelSize;
  }
  set kernelSize(value) {
    this.blurPass.kernelSize = value;
  }
  get distinction() {
    console.warn(this.name, "The distinction field has been removed, use luminanceMaterial.threshold and luminanceMaterial.smoothing instead.");
    return 1;
  }
  set distinction(value) {
    console.warn(this.name, "The distinction field has been removed, use luminanceMaterial.threshold and luminanceMaterial.smoothing instead.");
  }
  get intensity() {
    return this.uniforms.get("intensity").value;
  }
  set intensity(value) {
    this.uniforms.get("intensity").value = value;
  }
  getResolutionScale() {
    return this.resolution.scale;
  }
  setResolutionScale(scale) {
    this.resolution.scale = scale;
  }
  update(renderer, inputBuffer, deltaTime) {
    const renderTarget = this.renderTarget;
    if (this.luminancePass.enabled) {
      this.luminancePass.render(renderer, inputBuffer, renderTarget);
      this.blurPass.render(renderer, renderTarget, renderTarget);
    } else {
      this.blurPass.render(renderer, inputBuffer, renderTarget);
    }
  }
  setSize(width, height) {
    this.blurPass.setSize(width, height);
    this.renderTarget.setSize(this.resolution.width, this.resolution.height);
  }
  initialize(renderer, alpha, frameBufferType) {
    this.blurPass.initialize(renderer, alpha, frameBufferType);
    if (!alpha && frameBufferType === UnsignedByteType9) {
      this.renderTarget.texture.format = RGBFormat5;
    }
    if (frameBufferType !== void 0) {
      this.renderTarget.texture.type = frameBufferType;
    }
  }
};

// src/effects/BokehEffect.js
import {Uniform as Uniform21} from "../build/three.module.js";

// src/effects/glsl/bokeh/shader.frag
var shader_default47 = "uniform float focus;uniform float dof;uniform float aperture;uniform float maxBlur;void mainImage(const in vec4 inputColor,const in vec2 uv,const in float depth,out vec4 outputColor){vec2 aspectCorrection=vec2(1.0,aspect);\n#ifdef PERSPECTIVE_CAMERA\nfloat viewZ=perspectiveDepthToViewZ(depth,cameraNear,cameraFar);float linearDepth=viewZToOrthographicDepth(viewZ,cameraNear,cameraFar);\n#else\nfloat linearDepth=depth;\n#endif\nfloat focusNear=clamp(focus-dof,0.0,1.0);float focusFar=clamp(focus+dof,0.0,1.0);float low=step(linearDepth,focusNear);float high=step(focusFar,linearDepth);float factor=(linearDepth-focusNear)*low+(linearDepth-focusFar)*high;vec2 dofBlur=vec2(clamp(factor*aperture,-maxBlur,maxBlur));vec2 dofblur9=dofBlur*0.9;vec2 dofblur7=dofBlur*0.7;vec2 dofblur4=dofBlur*0.4;vec4 color=inputColor;color+=texture2D(inputBuffer,uv+(vec2(0.0,0.4)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.15,0.37)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.29,0.29)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.37,0.15)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.40,0.0)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.37,-0.15)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.29,-0.29)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.15,-0.37)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.0,-0.4)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.15,0.37)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.29,0.29)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.37,0.15)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.4,0.0)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.37,-0.15)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(-0.29,-0.29)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.15,-0.37)*aspectCorrection)*dofBlur);color+=texture2D(inputBuffer,uv+(vec2(0.15,0.37)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(-0.37,0.15)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(0.37,-0.15)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(-0.15,-0.37)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(-0.15,0.37)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(0.37,0.15)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(-0.37,-0.15)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(0.15,-0.37)*aspectCorrection)*dofblur9);color+=texture2D(inputBuffer,uv+(vec2(0.29,0.29)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(0.40,0.0)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(0.29,-0.29)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(0.0,-0.4)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(-0.29,0.29)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(-0.4,0.0)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(-0.29,-0.29)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(0.0,0.4)*aspectCorrection)*dofblur7);color+=texture2D(inputBuffer,uv+(vec2(0.29,0.29)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(0.4,0.0)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(0.29,-0.29)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(0.0,-0.4)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(-0.29,0.29)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(-0.4,0.0)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(-0.29,-0.29)*aspectCorrection)*dofblur4);color+=texture2D(inputBuffer,uv+(vec2(0.0,0.4)*aspectCorrection)*dofblur4);outputColor=color/41.0;}";

// src/effects/BokehEffect.js
var BokehEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, focus = 0.5, dof = 0.02, aperture = 0.015, maxBlur = 1} = {}) {
    super("BokehEffect", shader_default47, {
      blendFunction,
      attributes: EffectAttribute.CONVOLUTION | EffectAttribute.DEPTH,
      uniforms: new Map([
        ["focus", new Uniform21(focus)],
        ["dof", new Uniform21(dof)],
        ["aperture", new Uniform21(aperture)],
        ["maxBlur", new Uniform21(maxBlur)]
      ])
    });
  }
};

// src/effects/BrightnessContrastEffect.js
import {Uniform as Uniform22} from "../build/three.module.js";

// src/effects/glsl/brightness-contrast/shader.frag
var shader_default48 = "uniform float brightness;uniform float contrast;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec3 color=inputColor.rgb+vec3(brightness-0.5);if(contrast>0.0){color/=vec3(1.0-contrast);}else{color*=vec3(1.0+contrast);}outputColor=vec4(min(color+vec3(0.5),1.0),inputColor.a);}";

// src/effects/BrightnessContrastEffect.js
var BrightnessContrastEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, brightness = 0, contrast = 0} = {}) {
    super("BrightnessContrastEffect", shader_default48, {
      blendFunction,
      uniforms: new Map([
        ["brightness", new Uniform22(brightness)],
        ["contrast", new Uniform22(contrast)]
      ])
    });
  }
};

// src/effects/glsl/color-average/shader.frag
var shader_default49 = "void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){float sum=inputColor.r+inputColor.g+inputColor.b;outputColor=vec4(vec3(sum/3.0),inputColor.a);}";

// src/effects/ColorAverageEffect.js
var ColorAverageEffect = class extends Effect {
  constructor(blendFunction = BlendFunction.NORMAL) {
    super("ColorAverageEffect", shader_default49, {blendFunction});
  }
};

// src/effects/ColorDepthEffect.js
import {Uniform as Uniform23} from "../build/three.module.js";

// src/effects/glsl/color-depth/shader.frag
var shader_default50 = "uniform float factor;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){outputColor=vec4(floor(inputColor.rgb*factor+0.5)/factor,inputColor.a);}";

// src/effects/ColorDepthEffect.js
var ColorDepthEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, bits = 16} = {}) {
    super("ColorDepthEffect", shader_default50, {
      blendFunction,
      uniforms: new Map([
        ["factor", new Uniform23(1)]
      ])
    });
    this.bits = 0;
    this.setBitDepth(bits);
  }
  getBitDepth() {
    return this.bits;
  }
  setBitDepth(bits) {
    this.bits = bits;
    this.uniforms.get("factor").value = Math.pow(2, bits / 3);
  }
};

// src/effects/ChromaticAberrationEffect.js
import {Uniform as Uniform24, Vector2 as Vector214} from "../build/three.module.js";

// src/effects/glsl/chromatic-aberration/shader.frag
var shader_default51 = "varying vec2 vUvR;varying vec2 vUvB;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec4 color=inputColor;\n#ifdef ALPHA\nvec2 ra=texture2D(inputBuffer,vUvR).ra;vec2 ba=texture2D(inputBuffer,vUvB).ba;color.r=ra.x;color.b=ba.x;color.a=max(max(ra.y,ba.y),inputColor.a);\n#else\ncolor.r=texture2D(inputBuffer,vUvR).r;color.b=texture2D(inputBuffer,vUvB).b;\n#endif\noutputColor=color;}";

// src/effects/glsl/chromatic-aberration/shader.vert
var shader_default52 = "uniform vec2 offset;varying vec2 vUvR;varying vec2 vUvB;void mainSupport(const in vec2 uv){vUvR=uv+offset;vUvB=uv-offset;}";

// src/effects/ChromaticAberrationEffect.js
var ChromaticAberrationEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, offset = new Vector214(1e-3, 5e-4)} = {}) {
    super("ChromaticAberrationEffect", shader_default51, {
      vertexShader: shader_default52,
      blendFunction,
      attributes: EffectAttribute.CONVOLUTION,
      uniforms: new Map([
        ["offset", new Uniform24(offset)]
      ])
    });
  }
  get offset() {
    return this.uniforms.get("offset").value;
  }
  set offset(value) {
    this.uniforms.get("offset").value = value;
  }
  initialize(renderer, alpha, frameBufferType) {
    if (alpha) {
      this.defines.set("ALPHA", "1");
    } else {
      this.defines.delete("ALPHA");
    }
  }
};

// src/effects/glsl/depth/shader.frag
var shader_default53 = "void mainImage(const in vec4 inputColor,const in vec2 uv,const in float depth,out vec4 outputColor){\n#ifdef INVERTED\nvec3 color=vec3(1.0-depth);\n#else\nvec3 color=vec3(depth);\n#endif\noutputColor=vec4(color,inputColor.a);}";

// src/effects/DepthEffect.js
var DepthEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, inverted = false} = {}) {
    super("DepthEffect", shader_default53, {
      blendFunction,
      attributes: EffectAttribute.DEPTH
    });
    this.inverted = inverted;
  }
  get inverted() {
    return this.defines.has("INVERTED");
  }
  set inverted(value) {
    if (this.inverted !== value) {
      if (value) {
        this.defines.set("INVERTED", "1");
      } else {
        this.defines.delete("INVERTED");
      }
      this.setChanged();
    }
  }
};

// src/effects/DepthOfFieldEffect.js
import {
  BasicDepthPacking as BasicDepthPacking4,
  LinearFilter as LinearFilter6,
  RGBFormat as RGBFormat6,
  Uniform as Uniform25,
  UnsignedByteType as UnsignedByteType10,
  WebGLRenderTarget as WebGLRenderTarget11
} from "../build/three.module.js";

// src/effects/glsl/depth-of-field/shader.frag
var shader_default54 = "#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D nearColorBuffer;uniform mediump sampler2D farColorBuffer;\n#else\nuniform lowp sampler2D nearColorBuffer;uniform lowp sampler2D farColorBuffer;\n#endif\nuniform lowp sampler2D nearCoCBuffer;uniform float scale;void mainImage(const in vec4 inputColor,const in vec2 uv,const in float depth,out vec4 outputColor){vec4 colorNear=texture2D(nearColorBuffer,uv);vec4 colorFar=texture2D(farColorBuffer,uv);float CoCNear=texture2D(nearCoCBuffer,uv).r;CoCNear=min(CoCNear*scale,1.0);vec4 result=inputColor*(1.0-colorFar.a)+colorFar;result=mix(result,colorNear,CoCNear);outputColor=result;}";

// src/effects/DepthOfFieldEffect.js
var DepthOfFieldEffect = class extends Effect {
  constructor(camera, {
    blendFunction = BlendFunction.NORMAL,
    focusDistance = 0,
    focalLength = 0.1,
    bokehScale = 1,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE
  } = {}) {
    super("DepthOfFieldEffect", shader_default54, {
      blendFunction,
      attributes: EffectAttribute.DEPTH,
      uniforms: new Map([
        ["nearColorBuffer", new Uniform25(null)],
        ["farColorBuffer", new Uniform25(null)],
        ["nearCoCBuffer", new Uniform25(null)],
        ["scale", new Uniform25(1)]
      ])
    });
    this.camera = camera;
    this.renderTarget = new WebGLRenderTarget11(1, 1, {
      minFilter: LinearFilter6,
      magFilter: LinearFilter6,
      stencilBuffer: false,
      depthBuffer: false
    });
    this.renderTarget.texture.name = "DoF.Intermediate";
    this.renderTarget.texture.generateMipmaps = false;
    this.renderTargetMasked = this.renderTarget.clone();
    this.renderTargetMasked.texture.name = "DoF.Masked.Far";
    this.renderTargetNear = this.renderTarget.clone();
    this.renderTargetNear.texture.name = "DoF.Bokeh.Near";
    this.uniforms.get("nearColorBuffer").value = this.renderTargetNear.texture;
    this.renderTargetFar = this.renderTarget.clone();
    this.renderTargetFar.texture.name = "DoF.Bokeh.Far";
    this.uniforms.get("farColorBuffer").value = this.renderTargetFar.texture;
    this.renderTargetCoC = this.renderTarget.clone();
    this.renderTargetCoC.texture.format = RGBFormat6;
    this.renderTargetCoC.texture.name = "DoF.CoC";
    this.renderTargetCoCBlurred = this.renderTargetCoC.clone();
    this.renderTargetCoCBlurred.texture.name = "DoF.CoC.Blurred";
    this.uniforms.get("nearCoCBuffer").value = this.renderTargetCoCBlurred.texture;
    this.cocPass = new ShaderPass(new CircleOfConfusionMaterial(camera));
    const cocMaterial = this.circleOfConfusionMaterial;
    cocMaterial.uniforms.focusDistance.value = focusDistance;
    cocMaterial.uniforms.focalLength.value = focalLength;
    this.blurPass = new BlurPass({width, height, kernelSize: KernelSize.MEDIUM});
    this.blurPass.resolution.resizable = this;
    this.maskPass = new ShaderPass(new MaskMaterial(this.renderTargetCoC.texture));
    const maskMaterial = this.maskPass.getFullscreenMaterial();
    maskMaterial.maskFunction = MaskFunction.MULTIPLY_RGB_SET_ALPHA;
    maskMaterial.colorChannel = ColorChannel.GREEN;
    this.bokehNearBasePass = new ShaderPass(new BokehMaterial(false, true));
    this.bokehNearFillPass = new ShaderPass(new BokehMaterial(true, true));
    this.bokehFarBasePass = new ShaderPass(new BokehMaterial(false, false));
    this.bokehFarFillPass = new ShaderPass(new BokehMaterial(true, false));
    this.bokehScale = bokehScale;
    this.target = null;
  }
  get circleOfConfusionMaterial() {
    return this.cocPass.getFullscreenMaterial();
  }
  get resolution() {
    return this.blurPass.resolution;
  }
  get bokehScale() {
    return this.uniforms.get("scale").value;
  }
  set bokehScale(value) {
    const passes = [
      this.bokehNearBasePass,
      this.bokehNearFillPass,
      this.bokehFarBasePass,
      this.bokehFarFillPass
    ];
    passes.map((p) => p.getFullscreenMaterial().uniforms.scale).forEach((u) => {
      u.value = value;
    });
    this.maskPass.getFullscreenMaterial().uniforms.strength.value = value;
    this.uniforms.get("scale").value = value;
  }
  calculateFocusDistance(target) {
    const camera = this.camera;
    const viewDistance = camera.far - camera.near;
    const distance = camera.position.distanceTo(target);
    return Math.min(Math.max(distance / viewDistance, 0), 1);
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking4) {
    const material = this.circleOfConfusionMaterial;
    material.uniforms.depthBuffer.value = depthTexture;
    material.depthPacking = depthPacking;
  }
  update(renderer, inputBuffer, deltaTime) {
    const renderTarget = this.renderTarget;
    const renderTargetCoC = this.renderTargetCoC;
    const renderTargetCoCBlurred = this.renderTargetCoCBlurred;
    const renderTargetMasked = this.renderTargetMasked;
    const bokehFarBasePass = this.bokehFarBasePass;
    const bokehFarFillPass = this.bokehFarFillPass;
    const farBaseUniforms = bokehFarBasePass.getFullscreenMaterial().uniforms;
    const farFillUniforms = bokehFarFillPass.getFullscreenMaterial().uniforms;
    const bokehNearBasePass = this.bokehNearBasePass;
    const bokehNearFillPass = this.bokehNearFillPass;
    const nearBaseUniforms = bokehNearBasePass.getFullscreenMaterial().uniforms;
    const nearFillUniforms = bokehNearFillPass.getFullscreenMaterial().uniforms;
    if (this.target !== null) {
      const distance = this.calculateFocusDistance(this.target);
      this.circleOfConfusionMaterial.uniforms.focusDistance.value = distance;
    }
    this.cocPass.render(renderer, null, renderTargetCoC);
    this.blurPass.render(renderer, renderTargetCoC, renderTargetCoCBlurred);
    this.maskPass.render(renderer, inputBuffer, renderTargetMasked);
    farBaseUniforms.cocBuffer.value = farFillUniforms.cocBuffer.value = renderTargetCoC.texture;
    bokehFarBasePass.render(renderer, renderTargetMasked, renderTarget);
    bokehFarFillPass.render(renderer, renderTarget, this.renderTargetFar);
    nearBaseUniforms.cocBuffer.value = nearFillUniforms.cocBuffer.value = renderTargetCoCBlurred.texture;
    bokehNearBasePass.render(renderer, inputBuffer, renderTarget);
    bokehNearFillPass.render(renderer, renderTarget, this.renderTargetNear);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    let resizables = [
      this.cocPass,
      this.blurPass,
      this.maskPass,
      this.bokehNearBasePass,
      this.bokehNearFillPass,
      this.bokehFarBasePass,
      this.bokehFarFillPass
    ];
    resizables.push(this.renderTargetCoC, this.renderTargetMasked);
    resizables.forEach((r) => r.setSize(width, height));
    const w = resolution.width;
    const h = resolution.height;
    resizables = [
      this.renderTarget,
      this.renderTargetNear,
      this.renderTargetFar,
      this.renderTargetCoCBlurred
    ];
    resizables.forEach((r) => r.setSize(w, h));
    const passes = [
      this.bokehNearBasePass,
      this.bokehNearFillPass,
      this.bokehFarBasePass,
      this.bokehFarFillPass
    ];
    passes.forEach((p) => p.getFullscreenMaterial().setTexelSize(1 / w, 1 / h));
  }
  initialize(renderer, alpha, frameBufferType) {
    const initializables = [
      this.cocPass,
      this.maskPass,
      this.bokehNearBasePass,
      this.bokehNearFillPass,
      this.bokehFarBasePass,
      this.bokehFarFillPass
    ];
    initializables.forEach((i) => i.initialize(renderer, alpha, frameBufferType));
    this.blurPass.initialize(renderer, alpha, UnsignedByteType10);
    if (!alpha && frameBufferType === UnsignedByteType10) {
      this.renderTargetNear.texture.type = RGBFormat6;
    }
    if (frameBufferType !== void 0) {
      this.renderTarget.texture.type = frameBufferType;
      this.renderTargetNear.texture.type = frameBufferType;
      this.renderTargetFar.texture.type = frameBufferType;
      this.renderTargetMasked.texture.type = frameBufferType;
    }
  }
};

// src/effects/DotScreenEffect.js
import {Uniform as Uniform26, Vector2 as Vector215} from "../build/three.module.js";

// src/effects/glsl/dot-screen/shader.frag
var shader_default55 = "uniform vec2 angle;uniform float scale;float pattern(const in vec2 uv){vec2 point=scale*vec2(dot(angle.yx,vec2(uv.x,-uv.y)),dot(angle,uv));return(sin(point.x)*sin(point.y))*4.0;}void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec3 color=vec3(inputColor.rgb*10.0-5.0+pattern(uv*resolution));outputColor=vec4(color,inputColor.a);}";

// src/effects/DotScreenEffect.js
var DotScreenEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, angle = Math.PI * 0.5, scale = 1} = {}) {
    super("DotScreenEffect", shader_default55, {
      blendFunction,
      uniforms: new Map([
        ["angle", new Uniform26(new Vector215())],
        ["scale", new Uniform26(scale)]
      ])
    });
    this.setAngle(angle);
  }
  setAngle(angle) {
    this.uniforms.get("angle").value.set(Math.sin(angle), Math.cos(angle));
  }
};

// src/effects/GammaCorrectionEffect.js
import {Uniform as Uniform27} from "../build/three.module.js";

// src/effects/glsl/gamma-correction/shader.frag
var shader_default56 = "uniform float gamma;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){outputColor=LinearToGamma(max(inputColor,0.0),gamma);}";

// src/effects/GammaCorrectionEffect.js
var GammaCorrectionEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, gamma = 2} = {}) {
    super("GammaCorrectionEffect", shader_default56, {
      blendFunction,
      uniforms: new Map([
        ["gamma", new Uniform27(gamma)]
      ])
    });
  }
};

// src/effects/GlitchEffect.js
import {
  NearestFilter as NearestFilter6,
  RepeatWrapping,
  RGBFormat as RGBFormat8,
  Uniform as Uniform28,
  Vector2 as Vector216
} from "../build/three.module.js";

// src/images/textures/NoiseTexture.js
import {
  DataTexture,
  LuminanceFormat as LuminanceFormat2,
  RedFormat,
  RGFormat,
  RGBFormat as RGBFormat7,
  RGBAFormat as RGBAFormat4,
  UnsignedByteType as UnsignedByteType11
} from "../build/three.module.js";
function getNoise(size, format, type) {
  const channels = new Map([
    [LuminanceFormat2, 1],
    [RedFormat, 1],
    [RGFormat, 2],
    [RGBFormat7, 3],
    [RGBAFormat4, 4]
  ]);
  let data;
  if (!channels.has(format)) {
    console.error("Invalid noise texture format");
  }
  if (type === UnsignedByteType11) {
    data = new Uint8Array(size * channels.get(format));
    for (let i = 0, l = data.length; i < l; ++i) {
      data[i] = Math.random() * 255;
    }
  } else {
    data = new Float32Array(size * channels.get(format));
    for (let i = 0, l = data.length; i < l; ++i) {
      data[i] = Math.random();
    }
  }
  return data;
}
var NoiseTexture = class extends DataTexture {
  constructor(width, height, format = LuminanceFormat2, type = UnsignedByteType11) {
    super(getNoise(width * height, format, type), width, height, format, type);
  }
};

// src/effects/glsl/glitch/shader.frag
var shader_default57 = "uniform lowp sampler2D perturbationMap;uniform bool active;uniform float columns;uniform float random;uniform vec2 seed;uniform vec2 distortion;void mainUv(inout vec2 uv){if(active){if(uv.y<distortion.x+columns&&uv.y>distortion.x-columns*random){float sx=clamp(ceil(seed.x),0.0,1.0);uv.y=sx*(1.0-(uv.y+distortion.y))+(1.0-sx)*distortion.y;}if(uv.x<distortion.y+columns&&uv.x>distortion.y-columns*random){float sy=clamp(ceil(seed.y),0.0,1.0);uv.x=sy*distortion.x+(1.0-sy)*(1.0-(uv.x+distortion.x));}vec2 normal=texture2D(perturbationMap,uv*random*random).rg;uv+=normal*seed*(random*0.2);}}";

// src/effects/GlitchEffect.js
var tag = "Glitch.Generated";
function randomFloat(low, high) {
  return low + Math.random() * (high - low);
}
var GlitchEffect = class extends Effect {
  constructor({
    blendFunction = BlendFunction.NORMAL,
    chromaticAberrationOffset = null,
    delay = new Vector216(1.5, 3.5),
    duration = new Vector216(0.6, 1),
    strength = new Vector216(0.3, 1),
    columns = 0.05,
    ratio = 0.85,
    perturbationMap = null,
    dtSize = 64
  } = {}) {
    super("GlitchEffect", shader_default57, {
      blendFunction,
      uniforms: new Map([
        ["perturbationMap", new Uniform28(null)],
        ["columns", new Uniform28(columns)],
        ["active", new Uniform28(false)],
        ["random", new Uniform28(1)],
        ["seed", new Uniform28(new Vector216())],
        ["distortion", new Uniform28(new Vector216())]
      ])
    });
    this.setPerturbationMap(perturbationMap === null ? this.generatePerturbationMap(dtSize) : perturbationMap);
    this.delay = delay;
    this.duration = duration;
    this.breakPoint = new Vector216(randomFloat(this.delay.x, this.delay.y), randomFloat(this.duration.x, this.duration.y));
    this.time = 0;
    this.seed = this.uniforms.get("seed").value;
    this.distortion = this.uniforms.get("distortion").value;
    this.mode = GlitchMode.SPORADIC;
    this.strength = strength;
    this.ratio = ratio;
    this.chromaticAberrationOffset = chromaticAberrationOffset;
  }
  get active() {
    return this.uniforms.get("active").value;
  }
  getPerturbationMap() {
    return this.uniforms.get("perturbationMap").value;
  }
  setPerturbationMap(map) {
    const currentMap = this.getPerturbationMap();
    if (currentMap !== null && currentMap.name === tag) {
      currentMap.dispose();
    }
    map.minFilter = map.magFilter = NearestFilter6;
    map.wrapS = map.wrapT = RepeatWrapping;
    map.generateMipmaps = false;
    this.uniforms.get("perturbationMap").value = map;
  }
  generatePerturbationMap(size = 64) {
    const map = new NoiseTexture(size, size, RGBFormat8);
    map.name = tag;
    return map;
  }
  update(renderer, inputBuffer, deltaTime) {
    const mode = this.mode;
    const breakPoint = this.breakPoint;
    const offset = this.chromaticAberrationOffset;
    const s = this.strength;
    let time = this.time;
    let active = false;
    let r = 0, a = 0;
    let trigger;
    if (mode !== GlitchMode.DISABLED) {
      if (mode === GlitchMode.SPORADIC) {
        time += deltaTime;
        trigger = time > breakPoint.x;
        if (time >= breakPoint.x + breakPoint.y) {
          breakPoint.set(randomFloat(this.delay.x, this.delay.y), randomFloat(this.duration.x, this.duration.y));
          time = 0;
        }
      }
      r = Math.random();
      this.uniforms.get("random").value = r;
      if (trigger && r > this.ratio || mode === GlitchMode.CONSTANT_WILD) {
        active = true;
        r *= s.y * 0.03;
        a = randomFloat(-Math.PI, Math.PI);
        this.seed.set(randomFloat(-s.y, s.y), randomFloat(-s.y, s.y));
        this.distortion.set(randomFloat(0, 1), randomFloat(0, 1));
      } else if (trigger || mode === GlitchMode.CONSTANT_MILD) {
        active = true;
        r *= s.x * 0.03;
        a = randomFloat(-Math.PI, Math.PI);
        this.seed.set(randomFloat(-s.x, s.x), randomFloat(-s.x, s.x));
        this.distortion.set(randomFloat(0, 1), randomFloat(0, 1));
      }
      this.time = time;
    }
    if (offset !== null) {
      if (active) {
        offset.set(Math.cos(a), Math.sin(a)).multiplyScalar(r);
      } else {
        offset.set(0, 0);
      }
    }
    this.uniforms.get("active").value = active;
  }
};
var GlitchMode = {
  DISABLED: 0,
  SPORADIC: 1,
  CONSTANT_MILD: 2,
  CONSTANT_WILD: 3
};

// src/effects/GodRaysEffect.js
import {
  BasicDepthPacking as BasicDepthPacking5,
  Color as Color4,
  DepthTexture as DepthTexture2,
  LinearFilter as LinearFilter7,
  Matrix4 as Matrix42,
  RGBFormat as RGBFormat9,
  Scene as Scene3,
  Uniform as Uniform29,
  Vector2 as Vector217,
  Vector3,
  UnsignedByteType as UnsignedByteType12,
  WebGLRenderTarget as WebGLRenderTarget12
} from "../build/three.module.js";

// src/effects/glsl/god-rays/shader.frag
var shader_default58 = "#ifdef FRAMEBUFFER_PRECISION_HIGH\nuniform mediump sampler2D map;\n#else\nuniform lowp sampler2D map;\n#endif\nvoid mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){outputColor=texture2D(map,uv);}";

// src/effects/GodRaysEffect.js
var v = new Vector3();
var m = new Matrix42();
var GodRaysEffect = class extends Effect {
  constructor(camera, lightSource, {
    blendFunction = BlendFunction.SCREEN,
    samples = 60,
    density = 0.96,
    decay = 0.9,
    weight = 0.4,
    exposure = 0.6,
    clampMax = 1,
    resolutionScale = 0.5,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    kernelSize = KernelSize.SMALL,
    blur = true
  } = {}) {
    super("GodRaysEffect", shader_default58, {
      blendFunction,
      attributes: EffectAttribute.DEPTH,
      uniforms: new Map([
        ["map", new Uniform29(null)]
      ])
    });
    this.camera = camera;
    this.lightSource = lightSource;
    this.lightSource.material.depthWrite = false;
    this.lightSource.material.transparent = true;
    this.lightScene = new Scene3();
    this.screenPosition = new Vector217();
    this.renderTargetA = new WebGLRenderTarget12(1, 1, {
      minFilter: LinearFilter7,
      magFilter: LinearFilter7,
      stencilBuffer: false,
      depthBuffer: false
    });
    this.renderTargetA.texture.name = "GodRays.Target.A";
    this.renderTargetB = this.renderTargetA.clone();
    this.renderTargetB.texture.name = "GodRays.Target.B";
    this.uniforms.get("map").value = this.renderTargetB.texture;
    this.renderTargetLight = this.renderTargetA.clone();
    this.renderTargetLight.texture.name = "GodRays.Light";
    this.renderTargetLight.depthBuffer = true;
    this.renderTargetLight.depthTexture = new DepthTexture2();
    this.renderPassLight = new RenderPass(this.lightScene, camera);
    this.renderPassLight.getClearPass().overrideClearColor = new Color4(0);
    this.clearPass = new ClearPass(true, false, false);
    this.clearPass.overrideClearColor = new Color4(0);
    this.blurPass = new BlurPass({resolutionScale, width, height, kernelSize});
    this.blurPass.resolution.resizable = this;
    this.depthMaskPass = new ShaderPass(new DepthMaskMaterial());
    const depthMaskMaterial = this.depthMaskPass.getFullscreenMaterial();
    depthMaskMaterial.uniforms.depthBuffer1.value = this.renderTargetLight.depthTexture;
    this.godRaysPass = new ShaderPass(new GodRaysMaterial(this.screenPosition));
    const godRaysMaterial = this.godRaysPass.getFullscreenMaterial();
    godRaysMaterial.uniforms.density.value = density;
    godRaysMaterial.uniforms.decay.value = decay;
    godRaysMaterial.uniforms.weight.value = weight;
    godRaysMaterial.uniforms.exposure.value = exposure;
    godRaysMaterial.uniforms.clampMax.value = clampMax;
    this.samples = samples;
    this.blur = blur;
  }
  get texture() {
    return this.renderTargetB.texture;
  }
  get godRaysMaterial() {
    return this.godRaysPass.getFullscreenMaterial();
  }
  get resolution() {
    return this.blurPass.resolution;
  }
  get width() {
    return this.resolution.width;
  }
  set width(value) {
    this.resolution.width = value;
  }
  get height() {
    return this.resolution.height;
  }
  set height(value) {
    this.resolution.height = value;
  }
  get dithering() {
    return this.godRaysMaterial.dithering;
  }
  set dithering(value) {
    const material = this.godRaysMaterial;
    material.dithering = value;
    material.needsUpdate = true;
  }
  get blur() {
    return this.blurPass.enabled;
  }
  set blur(value) {
    this.blurPass.enabled = value;
  }
  get kernelSize() {
    return this.blurPass.kernelSize;
  }
  set kernelSize(value) {
    this.blurPass.kernelSize = value;
  }
  getResolutionScale() {
    return this.resolution.scale;
  }
  setResolutionScale(scale) {
    this.resolution.scale = scale;
  }
  get samples() {
    return this.godRaysMaterial.samples;
  }
  set samples(value) {
    this.godRaysMaterial.samples = value;
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking5) {
    const material = this.depthMaskPass.getFullscreenMaterial();
    material.uniforms.depthBuffer0.value = depthTexture;
    material.defines.DEPTH_PACKING_0 = depthPacking.toFixed(0);
    material.needsUpdate = true;
  }
  update(renderer, inputBuffer, deltaTime) {
    const lightSource = this.lightSource;
    const parent = lightSource.parent;
    const matrixAutoUpdate = lightSource.matrixAutoUpdate;
    const renderTargetA = this.renderTargetA;
    const renderTargetLight = this.renderTargetLight;
    lightSource.material.depthWrite = true;
    lightSource.matrixAutoUpdate = false;
    lightSource.updateWorldMatrix(true, false);
    if (parent !== null) {
      if (!matrixAutoUpdate) {
        m.copy(lightSource.matrix);
      }
      lightSource.matrix.copy(lightSource.matrixWorld);
    }
    this.lightScene.add(lightSource);
    this.renderPassLight.render(renderer, renderTargetLight);
    this.clearPass.render(renderer, renderTargetA);
    this.depthMaskPass.render(renderer, renderTargetLight, renderTargetA);
    lightSource.material.depthWrite = false;
    lightSource.matrixAutoUpdate = matrixAutoUpdate;
    if (parent !== null) {
      if (!matrixAutoUpdate) {
        lightSource.matrix.copy(m);
      }
      parent.add(lightSource);
    }
    v.setFromMatrixPosition(lightSource.matrixWorld).project(this.camera);
    this.screenPosition.set(Math.min(Math.max((v.x + 1) * 0.5, -1), 2), Math.min(Math.max((v.y + 1) * 0.5, -1), 2));
    if (this.blur) {
      this.blurPass.render(renderer, renderTargetA, renderTargetA);
    }
    this.godRaysPass.render(renderer, renderTargetA, this.renderTargetB);
  }
  setSize(width, height) {
    this.blurPass.setSize(width, height);
    this.renderPassLight.setSize(width, height);
    this.depthMaskPass.setSize(width, height);
    this.godRaysPass.setSize(width, height);
    const w = this.resolution.width;
    const h = this.resolution.height;
    this.renderTargetA.setSize(w, h);
    this.renderTargetB.setSize(w, h);
    this.renderTargetLight.setSize(w, h);
  }
  initialize(renderer, alpha, frameBufferType) {
    this.blurPass.initialize(renderer, alpha, frameBufferType);
    this.renderPassLight.initialize(renderer, alpha, frameBufferType);
    this.depthMaskPass.initialize(renderer, alpha, frameBufferType);
    this.godRaysPass.initialize(renderer, alpha, frameBufferType);
    if (!alpha && frameBufferType === UnsignedByteType12) {
      this.renderTargetA.texture.format = RGBFormat9;
      this.renderTargetB.texture.format = RGBFormat9;
      this.renderTargetLight.texture.format = RGBFormat9;
    }
    if (frameBufferType !== void 0) {
      this.renderTargetA.texture.type = frameBufferType;
      this.renderTargetB.texture.type = frameBufferType;
      this.renderTargetLight.texture.type = frameBufferType;
    }
  }
};

// src/effects/GridEffect.js
import {Uniform as Uniform30, Vector2 as Vector218} from "../build/three.module.js";

// src/effects/glsl/grid/shader.frag
var shader_default59 = "uniform vec2 scale;uniform float lineWidth;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){float grid=0.5-max(abs(mod(uv.x*scale.x,1.0)-0.5),abs(mod(uv.y*scale.y,1.0)-0.5));outputColor=vec4(vec3(smoothstep(0.0,lineWidth,grid)),inputColor.a);}";

// src/effects/GridEffect.js
var GridEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.OVERLAY, scale = 1, lineWidth = 0} = {}) {
    super("GridEffect", shader_default59, {
      blendFunction,
      uniforms: new Map([
        ["scale", new Uniform30(new Vector218())],
        ["lineWidth", new Uniform30(lineWidth)]
      ])
    });
    this.resolution = new Vector218();
    this.scale = Math.max(scale, 1e-6);
    this.lineWidth = Math.max(lineWidth, 0);
  }
  getScale() {
    return this.scale;
  }
  setScale(scale) {
    this.scale = scale;
    this.setSize(this.resolution.x, this.resolution.y);
  }
  getLineWidth() {
    return this.lineWidth;
  }
  setLineWidth(lineWidth) {
    this.lineWidth = lineWidth;
    this.setSize(this.resolution.x, this.resolution.y);
  }
  setSize(width, height) {
    this.resolution.set(width, height);
    const aspect = width / height;
    const scale = this.scale * (height * 0.125);
    this.uniforms.get("scale").value.set(aspect * scale, scale);
    this.uniforms.get("lineWidth").value = scale / height + this.lineWidth;
  }
};

// src/effects/HueSaturationEffect.js
import {Uniform as Uniform31, Vector3 as Vector32} from "../build/three.module.js";

// src/effects/glsl/hue-saturation/shader.frag
var shader_default60 = "uniform vec3 hue;uniform float saturation;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec3 color=vec3(dot(inputColor.rgb,hue.xyz),dot(inputColor.rgb,hue.zxy),dot(inputColor.rgb,hue.yzx));float average=(color.r+color.g+color.b)/3.0;vec3 diff=average-color;if(saturation>0.0){color+=diff*(1.0-1.0/(1.001-saturation));}else{color+=diff*-saturation;}outputColor=vec4(min(color,1.0),inputColor.a);}";

// src/effects/HueSaturationEffect.js
var HueSaturationEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, hue = 0, saturation = 0} = {}) {
    super("HueSaturationEffect", shader_default60, {
      blendFunction,
      uniforms: new Map([
        ["hue", new Uniform31(new Vector32())],
        ["saturation", new Uniform31(saturation)]
      ])
    });
    this.setHue(hue);
  }
  setHue(hue) {
    const s = Math.sin(hue), c2 = Math.cos(hue);
    this.uniforms.get("hue").value.set(2 * c2, -Math.sqrt(3) * s - c2, Math.sqrt(3) * s - c2).addScalar(1).divideScalar(3);
  }
};

// src/effects/LUTEffect.js
import {
  DataTexture3D as DataTexture3D2,
  FloatType as FloatType5,
  HalfFloatType as HalfFloatType2,
  LinearEncoding as LinearEncoding2,
  LinearFilter as LinearFilter9,
  NearestFilter as NearestFilter7,
  sRGBEncoding as sRGBEncoding2,
  Uniform as Uniform32,
  Vector3 as Vector34
} from "../build/three.module.js";

// src/images/textures/LookupTexture3D.js
import {
  Color as Color5,
  ClampToEdgeWrapping,
  DataTexture as DataTexture2,
  DataTexture3D,
  FloatType as FloatType4,
  LinearFilter as LinearFilter8,
  LinearEncoding,
  RGBFormat as RGBFormat10,
  RGBAFormat as RGBAFormat5,
  sRGBEncoding,
  UnsignedByteType as UnsignedByteType13,
  Vector3 as Vector33
} from "../build/three.module.js";

// src/images/RawImageData.js
function createCanvas(width, height, data) {
  const canvas = document.createElementNS("http://www.w3.org/1999/xhtml", "canvas");
  const context = canvas.getContext("2d");
  canvas.width = width;
  canvas.height = height;
  if (data instanceof Image) {
    context.drawImage(data, 0, 0);
  } else {
    const imageData = context.createImageData(width, height);
    imageData.data.set(data);
    context.putImageData(imageData, 0, 0);
  }
  return canvas;
}
var RawImageData = class {
  constructor(width = 0, height = 0, data = null) {
    this.width = width;
    this.height = height;
    this.data = data;
  }
  toCanvas() {
    return typeof document === "undefined" ? null : createCanvas(this.width, this.height, this.data);
  }
  static from(image) {
    const {width, height} = image;
    let data;
    if (image instanceof Image) {
      const canvas = createCanvas(width, height, image);
      if (canvas !== null) {
        const context = canvas.getContext("2d");
        data = context.getImageData(0, 0, width, height).data;
      }
    } else {
      data = image.data;
    }
    return new RawImageData(width, height, data);
  }
};

// src/images/lut/LUTOperation.js
var LUTOperation = {
  SCALE_UP: "lut.scaleup"
};

// tmp/lut/worker.txt
var worker_default = '(()=>{var q={SCALE_UP:"lut.scaleup"};var _=[new Float32Array(3),new Float32Array(3)],t=[new Float32Array(3),new Float32Array(3),new Float32Array(3),new Float32Array(3)],U=[[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([1,0,0]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([1,0,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([1,1,0]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,1,0]),new Float32Array([0,1,1]),new Float32Array([1,1,1])],[new Float32Array([0,0,0]),new Float32Array([0,0,1]),new Float32Array([0,1,1]),new Float32Array([1,1,1])]];function L(a,n,r,m){let h=r[0]-n[0],e=r[1]-n[1],s=r[2]-n[2],l=a[0]-n[0],w=a[1]-n[1],c=a[2]-n[2],y=e*c-s*w,A=s*l-h*c,g=h*w-e*l,p=Math.sqrt(y*y+A*A+g*g),V=p*.5,F=y/p,f=A/p,i=g/p,u=-(a[0]*F+a[1]*f+a[2]*i),M=m[0]*F+m[1]*f+m[2]*i;return Math.abs(M+u)*V/3}function X(a,n,r,m,h,e){let s=(r+m*n+h*n*n)*3;e[0]=a[s+0],e[1]=a[s+1],e[2]=a[s+2]}function k(a,n,r,m,h,e){let s=r*(n-1),l=m*(n-1),w=h*(n-1),c=Math.floor(s),y=Math.floor(l),A=Math.floor(w),g=Math.ceil(s),p=Math.ceil(l),V=Math.ceil(w),F=s-c,f=l-y,i=w-A;if(c===s&&y===l&&A===w)X(a,n,s,l,w,e);else{let u;F>=f&&f>=i?u=U[0]:F>=i&&i>=f?u=U[1]:i>=F&&F>=f?u=U[2]:f>=F&&F>=i?u=U[3]:f>=i&&i>=F?u=U[4]:i>=f&&f>=F&&(u=U[5]);let[M,x,P,T]=u,d=_[0];d[0]=F,d[1]=f,d[2]=i;let o=_[1],Y=g-c,Z=p-y,b=V-A;o[0]=Y*M[0]+c,o[1]=Z*M[1]+y,o[2]=b*M[2]+A,X(a,n,o[0],o[1],o[2],t[0]),o[0]=Y*x[0]+c,o[1]=Z*x[1]+y,o[2]=b*x[2]+A,X(a,n,o[0],o[1],o[2],t[1]),o[0]=Y*P[0]+c,o[1]=Z*P[1]+y,o[2]=b*P[2]+A,X(a,n,o[0],o[1],o[2],t[2]),o[0]=Y*T[0]+c,o[1]=Z*T[1]+y,o[2]=b*T[2]+A,X(a,n,o[0],o[1],o[2],t[3]);let v=L(x,P,T,d)*6,S=L(M,P,T,d)*6,C=L(M,x,T,d)*6,E=L(M,x,P,d)*6;t[0][0]*=v,t[0][1]*=v,t[0][2]*=v,t[1][0]*=S,t[1][1]*=S,t[1][2]*=S,t[2][0]*=C,t[2][1]*=C,t[2][2]*=C,t[3][0]*=E,t[3][1]*=E,t[3][2]*=E,e[0]=t[0][0]+t[1][0]+t[2][0]+t[3][0],e[1]=t[0][1]+t[1][1]+t[2][1]+t[3][1],e[2]=t[0][2]+t[1][2]+t[2][2]+t[3][2]}}var O=class{static expand(n,r){let m=Math.cbrt(n.length/3),h=new Float32Array(3),e=new n.constructor(r**3*3),s=1/(r-1);for(let l=0;l<r;++l)for(let w=0;w<r;++w)for(let c=0;c<r;++c){let y=c*s,A=w*s,g=l*s,p=Math.round(c+w*r+l*r*r)*3;k(n,m,y,A,g,h),e[p+0]=h[0],e[p+1]=h[1],e[p+2]=h[2]}return e}};self.addEventListener("message",a=>{let n=a.data,r=n.data;switch(n.operation){case q.SCALE_UP:r=O.expand(r,n.size);break}postMessage(r,[r.buffer]),close()});})();\n';

// src/images/textures/LookupTexture3D.js
var c = new Color5();
var LookupTexture3D = class extends DataTexture3D {
  constructor(data, size) {
    super(data, size, size, size);
    this.type = FloatType4;
    this.format = RGBFormat10;
    this.encoding = LinearEncoding;
    this.minFilter = LinearFilter8;
    this.magFilter = LinearFilter8;
    this.wrapS = ClampToEdgeWrapping;
    this.wrapT = ClampToEdgeWrapping;
    this.wrapR = ClampToEdgeWrapping;
    this.unpackAlignment = 1;
    this.domainMin = new Vector33(0, 0, 0);
    this.domainMax = new Vector33(1, 1, 1);
  }
  get isLookupTexture3D() {
    return true;
  }
  scaleUp(size, transferData = true) {
    const image = this.image;
    let promise;
    if (size <= image.width) {
      promise = Promise.reject(new Error("The target size must be greater than the current size"));
    } else {
      const workerURL = URL.createObjectURL(new Blob([worker_default], {type: "text/javascript"}));
      const worker = new Worker(workerURL);
      promise = new Promise((resolve, reject) => {
        worker.addEventListener("error", (event) => reject(event.error));
        worker.addEventListener("message", (event) => {
          const lut = new LookupTexture3D(event.data, size);
          lut.encoding = this.encoding;
          lut.type = this.type;
          lut.name = this.name;
          URL.revokeObjectURL(workerURL);
          resolve(lut);
        });
        const transferList = transferData ? [image.data.buffer] : [];
        worker.postMessage({
          operation: LUTOperation.SCALE_UP,
          data: image.data,
          size
        }, transferList);
      });
    }
    return promise;
  }
  applyLUT(lut) {
    const img0 = this.image;
    const img1 = lut.image;
    const size0 = Math.min(img0.width, img0.height, img0.depth);
    const size1 = Math.min(img1.width, img1.height, img1.depth);
    if (size0 !== size1) {
      console.error("Size mismatch");
    } else if (lut.type !== FloatType4 || this.type !== FloatType4) {
      console.error("Both LUTs must be FloatType textures");
    } else if (lut.format !== RGBFormat10 || this.format !== RGBFormat10) {
      console.error("Both LUTs must be RGB textures");
    } else {
      const data0 = img0.data;
      const data1 = img1.data;
      const size = size0;
      const s = size - 1;
      for (let i = 0, l = size ** 3; i < l; ++i) {
        const i3 = i * 3;
        const r = data0[i3 + 0] * s;
        const g = data0[i3 + 1] * s;
        const b = data0[i3 + 2] * s;
        const iRGB = Math.round(r + g * size + b * size * size) * 3;
        data0[i3 + 0] = data1[iRGB + 0];
        data0[i3 + 1] = data1[iRGB + 1];
        data0[i3 + 2] = data1[iRGB + 2];
      }
      this.needsUpdate = true;
    }
    return this;
  }
  convertToUint8() {
    if (this.type === FloatType4) {
      const floatData = this.image.data;
      const uint8Data = new Uint8ClampedArray(floatData.length);
      for (let i = 0, l = floatData.length; i < l; ++i) {
        uint8Data[i] = floatData[i] * 255;
      }
      this.image.data = uint8Data;
      this.type = UnsignedByteType13;
      this.needsUpdate = true;
    }
    return this;
  }
  convertToFloat() {
    if (this.type === UnsignedByteType13) {
      const uint8Data = this.image.data;
      const floatData = new Float32Array(uint8Data.length);
      for (let i = 0, l = uint8Data.length; i < l; ++i) {
        floatData[i] = uint8Data[i] / 255;
      }
      this.image.data = floatData;
      this.type = FloatType4;
      this.needsUpdate = true;
    }
    return this;
  }
  convertLinearToSRGB() {
    const data = this.image.data;
    if (this.type === FloatType4) {
      const stride = this.format === RGBAFormat5 ? 4 : 3;
      for (let i = 0, l = data.length; i < l; i += stride) {
        c.fromArray(data, i).convertLinearToSRGB().toArray(data, i);
      }
      this.encoding = sRGBEncoding;
      this.needsUpdate = true;
    } else {
      console.error("Color space conversion requires FloatType data");
    }
    return this;
  }
  convertSRGBToLinear() {
    const data = this.image.data;
    if (this.type === FloatType4) {
      const stride = this.format === RGBAFormat5 ? 4 : 3;
      for (let i = 0, l = data.length; i < l; i += stride) {
        c.fromArray(data, i).convertSRGBToLinear().toArray(data, i);
      }
      this.encoding = LinearEncoding;
      this.needsUpdate = true;
    } else {
      console.error("Color space conversion requires FloatType data");
    }
    return this;
  }
  convertToRGBA() {
    if (this.format === RGBFormat10) {
      const size = this.image.width;
      const rgbData = this.image.data;
      const rgbaData = new rgbData.constructor(size ** 3 * 4);
      const maxValue = this.type === FloatType4 ? 1 : 255;
      for (let i = 0, j = 0, l = rgbData.length; i < l; i += 3, j += 4) {
        rgbaData[j + 0] = rgbData[i + 0];
        rgbaData[j + 1] = rgbData[i + 1];
        rgbaData[j + 2] = rgbData[i + 2];
        rgbaData[j + 3] = maxValue;
      }
      this.image.data = rgbaData;
      this.format = RGBAFormat5;
      this.needsUpdate = true;
    }
    return this;
  }
  toDataTexture() {
    const width = this.image.width;
    const height = this.image.height * this.image.depth;
    const texture = new DataTexture2(this.image.data, width, height);
    texture.name = this.name;
    texture.type = this.type;
    texture.format = this.format;
    texture.encoding = this.encoding;
    texture.minFilter = LinearFilter8;
    texture.magFilter = LinearFilter8;
    texture.wrapS = this.wrapS;
    texture.wrapT = this.wrapT;
    texture.generateMipmaps = false;
    return texture;
  }
  static from(texture) {
    const image = texture.image;
    const {width, height} = image;
    const size = Math.min(width, height);
    let data;
    if (image instanceof Image) {
      const rawImageData = RawImageData.from(image);
      data = rawImageData.data;
      const rearrangedData = new Uint8Array(size ** 3 * 3);
      if (width > height) {
        for (let z = 0; z < size; ++z) {
          for (let y = 0; y < size; ++y) {
            for (let x = 0; x < size; ++x) {
              const i4 = (x + z * size + y * size * size) * 4;
              const i3 = (x + y * size + z * size * size) * 3;
              rearrangedData[i3 + 0] = data[i4 + 0];
              rearrangedData[i3 + 1] = data[i4 + 1];
              rearrangedData[i3 + 2] = data[i4 + 2];
            }
          }
        }
      } else {
        for (let i = 0, l = size ** 3; i < l; ++i) {
          const i4 = i * 4;
          const i3 = i * 3;
          rearrangedData[i3 + 0] = data[i4 + 0];
          rearrangedData[i3 + 1] = data[i4 + 1];
          rearrangedData[i3 + 2] = data[i4 + 2];
        }
      }
      data = rearrangedData;
    } else {
      data = image.data.slice();
    }
    const lut = new LookupTexture3D(data, size);
    lut.type = texture.type;
    lut.encoding = texture.encoding;
    lut.name = texture.name;
    return lut;
  }
  static createNeutral(size) {
    const data = new Float32Array(size ** 3 * 3);
    const s = 1 / (size - 1);
    for (let r = 0; r < size; ++r) {
      for (let g = 0; g < size; ++g) {
        for (let b = 0; b < size; ++b) {
          const i3 = (r + g * size + b * size * size) * 3;
          data[i3 + 0] = r * s;
          data[i3 + 1] = g * s;
          data[i3 + 2] = b * s;
        }
      }
    }
    const lut = new LookupTexture3D(data, size);
    lut.name = "neutral";
    return lut;
  }
};

// src/effects/glsl/lut/shader.frag
var shader_default61 = "uniform vec3 scale;uniform vec3 offset;\n#ifdef CUSTOM_INPUT_DOMAIN\nuniform vec3 domainMin;uniform vec3 domainMax;\n#endif\n#ifdef LUT_3D\n#ifdef LUT_PRECISION_HIGH\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler3D lut;\n#else\nuniform mediump sampler3D lut;\n#endif\n#else\nuniform lowp sampler3D lut;\n#endif\nvec4 applyLUT(const in vec3 rgb){\n#ifdef TETRAHEDRAL_INTERPOLATION\nvec3 p=floor(rgb);vec3 f=rgb-p;vec3 v1=(p+0.5)*LUT_TEXEL_WIDTH;vec3 v4=(p+1.5)*LUT_TEXEL_WIDTH;vec3 v2,v3;vec3 frac;if(f.r>=f.g){if(f.g>f.b){frac=f.rgb;v2=vec3(v4.x,v1.y,v1.z);v3=vec3(v4.x,v4.y,v1.z);}else if(f.r>=f.b){frac=f.rbg;v2=vec3(v4.x,v1.y,v1.z);v3=vec3(v4.x,v1.y,v4.z);}else{frac=f.brg;v2=vec3(v1.x,v1.y,v4.z);v3=vec3(v4.x,v1.y,v4.z);}}else{if(f.b>f.g){frac=f.bgr;v2=vec3(v1.x,v1.y,v4.z);v3=vec3(v1.x,v4.y,v4.z);}else if(f.r>=f.b){frac=f.grb;v2=vec3(v1.x,v4.y,v1.z);v3=vec3(v4.x,v4.y,v1.z);}else{frac=f.gbr;v2=vec3(v1.x,v4.y,v1.z);v3=vec3(v1.x,v4.y,v4.z);}}vec4 n1=texture(lut,v1);vec4 n2=texture(lut,v2);vec4 n3=texture(lut,v3);vec4 n4=texture(lut,v4);vec4 weights=vec4(1.0-frac.x,frac.x-frac.y,frac.y-frac.z,frac.z);vec4 result=weights*mat4(vec4(n1.r,n2.r,n3.r,n4.r),vec4(n1.g,n2.g,n3.g,n4.g),vec4(n1.b,n2.b,n3.b,n4.b),vec4(1.0));return vec4(result.rgb,1.0);\n#else\nreturn texture(lut,rgb);\n#endif\n}\n#else\n#ifdef LUT_PRECISION_HIGH\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D lut;\n#else\nuniform mediump sampler2D lut;\n#endif\n#else\nuniform lowp sampler2D lut;\n#endif\nvec4 applyLUT(const in vec3 rgb){float slice=rgb.b*LUT_SIZE;float slice0=floor(slice);float interp=slice-slice0;float centeredInterp=interp-0.5;float slice1=slice0+sign(centeredInterp);\n#ifdef LUT_STRIP_HORIZONTAL\nfloat xOffset=clamp(rgb.r*LUT_TEXEL_HEIGHT,LUT_TEXEL_WIDTH*0.5,LUT_TEXEL_HEIGHT-LUT_TEXEL_WIDTH*0.5);vec2 uv0=vec2(slice0*LUT_TEXEL_HEIGHT+xOffset,rgb.g);vec2 uv1=vec2(slice1*LUT_TEXEL_HEIGHT+xOffset,rgb.g);\n#else\nfloat yOffset=clamp(rgb.g*LUT_TEXEL_WIDTH,LUT_TEXEL_HEIGHT*0.5,LUT_TEXEL_WIDTH-LUT_TEXEL_HEIGHT*0.5);vec2 uv0=vec2(rgb.r,slice0*LUT_TEXEL_WIDTH+yOffset);vec2 uv1=vec2(rgb.r,slice1*LUT_TEXEL_WIDTH+yOffset);\n#endif\nvec4 sample0=texture2D(lut,uv0);vec4 sample1=texture2D(lut,uv1);return mix(sample0,sample1,abs(centeredInterp));}\n#endif\nvoid mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec3 c=linearToInputTexel(inputColor).rgb;\n#ifdef CUSTOM_INPUT_DOMAIN\nif(c.r>=domainMin.r&&c.g>=domainMin.g&&c.b>=domainMin.b&&c.r<=domainMax.r&&c.g<=domainMax.g&&c.b<=domainMax.b){c=texelToLinear(applyLUT(scale*c+offset)).rgb;}else{c=inputColor.rgb;}\n#else\n#if !defined(LUT_3D) || defined(TETRAHEDRAL_INTERPOLATION)\nc=clamp(c,0.0,1.0);\n#endif\nc=texelToLinear(applyLUT(scale*c+offset)).rgb;\n#endif\noutputColor=vec4(c,inputColor.a);}";

// src/effects/LUTEffect.js
var LUTEffect = class extends Effect {
  constructor(lut, {
    blendFunction = BlendFunction.NORMAL,
    tetrahedralInterpolation = false
  } = {}) {
    super("LUTEffect", shader_default61, {
      blendFunction,
      uniforms: new Map([
        ["lut", new Uniform32(null)],
        ["scale", new Uniform32(new Vector34())],
        ["offset", new Uniform32(new Vector34())],
        ["domainMin", new Uniform32(null)],
        ["domainMax", new Uniform32(null)]
      ])
    });
    this.tetrahedralInterpolation = tetrahedralInterpolation;
    this.inputEncoding = sRGBEncoding2;
    this.outputEncoding = this.inputEncoding;
    this.setInputEncoding(sRGBEncoding2);
    this.setLUT(lut);
  }
  getOutputEncoding() {
    return this.outputEncoding;
  }
  getInputEncoding() {
    return this.inputEncoding;
  }
  setInputEncoding(value) {
    const defines = this.defines;
    const lut = this.getLUT();
    switch (value) {
      case sRGBEncoding2:
        defines.set("linearToInputTexel(texel)", "LinearTosRGB(texel)");
        break;
      case LinearEncoding2:
        defines.set("linearToInputTexel(texel)", "texel");
        break;
      default:
        console.error("Unsupported input encoding:", value);
        break;
    }
    if (lut !== null) {
      this.outputEncoding = lut.encoding === LinearEncoding2 ? value : lut.encoding;
      switch (this.outputEncoding) {
        case sRGBEncoding2:
          defines.set("texelToLinear(texel)", "sRGBToLinear(texel)");
          break;
        case LinearEncoding2:
          defines.set("texelToLinear(texel)", "texel");
          break;
        default:
          console.error("Unsupported LUT encoding:", lut.encoding);
          break;
      }
    }
    if (this.inputEncoding !== value) {
      this.inputEncoding = value;
      this.setChanged();
    }
  }
  getLUT() {
    return this.uniforms.get("lut").value;
  }
  setLUT(lut) {
    const defines = this.defines;
    const uniforms = this.uniforms;
    if (this.getLUT() !== lut) {
      const image = lut.image;
      defines.clear();
      defines.set("LUT_SIZE", Math.min(image.width, image.height).toFixed(16));
      defines.set("LUT_TEXEL_WIDTH", (1 / image.width).toFixed(16));
      defines.set("LUT_TEXEL_HEIGHT", (1 / image.height).toFixed(16));
      uniforms.get("lut").value = lut;
      uniforms.get("domainMin").value = null;
      uniforms.get("domainMax").value = null;
      if (lut.type === FloatType5 || lut.type === HalfFloatType2) {
        defines.set("LUT_PRECISION_HIGH", "1");
      }
      if (image.width > image.height) {
        defines.set("LUT_STRIP_HORIZONTAL", "1");
      } else if (lut instanceof DataTexture3D2) {
        defines.set("LUT_3D", "1");
      }
      if (lut instanceof LookupTexture3D) {
        const min = lut.domainMin;
        const max = lut.domainMax;
        if (min.x !== 0 || min.y !== 0 || min.z !== 0 || max.x !== 1 || max.y !== 1 || max.z !== 1) {
          defines.set("CUSTOM_INPUT_DOMAIN", "1");
          uniforms.get("domainMin").value = min.clone();
          uniforms.get("domainMax").value = max.clone();
        }
      }
      this.configureTetrahedralInterpolation();
      this.updateScaleOffset();
      this.setInputEncoding(this.inputEncoding);
      this.setChanged();
    }
  }
  updateScaleOffset() {
    const lut = this.getLUT();
    const size = Math.min(lut.image.width, lut.image.height);
    const scale = this.uniforms.get("scale").value;
    const offset = this.uniforms.get("offset").value;
    if (this.defines.has("TETRAHEDRAL_INTERPOLATION")) {
      if (this.defines.has("CUSTOM_INPUT_DOMAIN")) {
        const domainScale = lut.domainMax.clone().sub(lut.domainMin);
        scale.setScalar(size - 1).divide(domainScale);
        offset.copy(lut.domainMin).negate().multiply(scale);
      } else {
        scale.setScalar(size - 1);
        offset.setScalar(0);
      }
    } else {
      if (this.defines.has("CUSTOM_INPUT_DOMAIN")) {
        const domainScale = lut.domainMax.clone().sub(lut.domainMin).multiplyScalar(size);
        scale.setScalar(size - 1).divide(domainScale);
        offset.copy(lut.domainMin).negate().multiply(scale).addScalar(1 / (2 * size));
      } else {
        scale.setScalar((size - 1) / size);
        offset.setScalar(1 / (2 * size));
      }
    }
  }
  configureTetrahedralInterpolation() {
    const lut = this.getLUT();
    lut.minFilter = LinearFilter9;
    lut.magFilter = LinearFilter9;
    this.defines.delete("TETRAHEDRAL_INTERPOLATION");
    if (this.tetrahedralInterpolation && lut !== null) {
      if (lut instanceof DataTexture3D2) {
        this.defines.set("TETRAHEDRAL_INTERPOLATION", "1");
        lut.minFilter = NearestFilter7;
        lut.magFilter = NearestFilter7;
      } else {
        console.warn("Tetrahedral interpolation requires a 3D texture");
      }
    }
    lut.needsUpdate = true;
  }
  setTetrahedralInterpolationEnabled(enabled) {
    this.tetrahedralInterpolation = enabled;
    this.configureTetrahedralInterpolation();
    this.updateScaleOffset();
    this.setChanged();
  }
};

// src/effects/glsl/noise/shader.frag
var shader_default62 = "void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec3 noise=vec3(rand(uv*time));\n#ifdef PREMULTIPLY\noutputColor=vec4(min(inputColor.rgb*noise,vec3(1.0)),inputColor.a);\n#else\noutputColor=vec4(noise,inputColor.a);\n#endif\n}";

// src/effects/NoiseEffect.js
var NoiseEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.SCREEN, premultiply = false} = {}) {
    super("NoiseEffect", shader_default62, {blendFunction});
    this.premultiply = premultiply;
  }
  get premultiply() {
    return this.defines.has("PREMULTIPLY");
  }
  set premultiply(value) {
    if (this.premultiply !== value) {
      if (value) {
        this.defines.set("PREMULTIPLY", "1");
      } else {
        this.defines.delete("PREMULTIPLY");
      }
      this.setChanged();
    }
  }
};

// src/effects/OutlineEffect.js
import {
  Color as Color6,
  LinearFilter as LinearFilter10,
  RepeatWrapping as RepeatWrapping2,
  RGBFormat as RGBFormat11,
  Uniform as Uniform33,
  UnsignedByteType as UnsignedByteType14,
  WebGLRenderTarget as WebGLRenderTarget13
} from "../build/three.module.js";

// src/effects/glsl/outline/shader.frag
var shader_default63 = "uniform lowp sampler2D edgeTexture;uniform lowp sampler2D maskTexture;uniform vec3 visibleEdgeColor;uniform vec3 hiddenEdgeColor;uniform float pulse;uniform float edgeStrength;\n#ifdef USE_PATTERN\nuniform lowp sampler2D patternTexture;varying vec2 vUvPattern;\n#endif\nvoid mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec2 edge=texture2D(edgeTexture,uv).rg;vec2 mask=texture2D(maskTexture,uv).rg;\n#ifndef X_RAY\nedge.y=0.0;\n#endif\nedge*=(edgeStrength*mask.x*pulse);vec3 color=edge.x*visibleEdgeColor+edge.y*hiddenEdgeColor;float visibilityFactor=0.0;\n#ifdef USE_PATTERN\nvec4 patternColor=texture2D(patternTexture,vUvPattern);\n#ifdef X_RAY\nfloat hiddenFactor=0.5;\n#else\nfloat hiddenFactor=0.0;\n#endif\nvisibilityFactor=(1.0-mask.y>0.0)? 1.0 : hiddenFactor;visibilityFactor*=(1.0-mask.x)*patternColor.a;color+=visibilityFactor*patternColor.rgb;\n#endif\nfloat alpha=max(max(edge.x,edge.y),visibilityFactor);\n#ifdef ALPHA\noutputColor=vec4(color,alpha);\n#else\noutputColor=vec4(color,max(alpha,inputColor.a));\n#endif\n}";

// src/effects/glsl/outline/shader.vert
var shader_default64 = "uniform float patternScale;varying vec2 vUvPattern;void mainSupport(const in vec2 uv){vUvPattern=uv*vec2(aspect,1.0)*patternScale;}";

// src/effects/OutlineEffect.js
var OutlineEffect = class extends Effect {
  constructor(scene, camera, {
    blendFunction = BlendFunction.SCREEN,
    patternTexture = null,
    edgeStrength = 1,
    pulseSpeed = 0,
    visibleEdgeColor = 16777215,
    hiddenEdgeColor = 2230538,
    resolutionScale = 0.5,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE,
    kernelSize = KernelSize.VERY_SMALL,
    blur = false,
    xRay = true
  } = {}) {
    super("OutlineEffect", shader_default63, {
      uniforms: new Map([
        ["maskTexture", new Uniform33(null)],
        ["edgeTexture", new Uniform33(null)],
        ["edgeStrength", new Uniform33(edgeStrength)],
        ["visibleEdgeColor", new Uniform33(new Color6(visibleEdgeColor))],
        ["hiddenEdgeColor", new Uniform33(new Color6(hiddenEdgeColor))],
        ["pulse", new Uniform33(1)],
        ["patternScale", new Uniform33(1)],
        ["patternTexture", new Uniform33(null)]
      ])
    });
    this.blendMode.addEventListener("change", (event) => {
      if (this.blendMode.getBlendFunction() === BlendFunction.ALPHA) {
        this.defines.set("ALPHA", "1");
      } else {
        this.defines.delete("ALPHA");
      }
      this.setChanged();
    });
    this.blendMode.setBlendFunction(blendFunction);
    this.setPatternTexture(patternTexture);
    this.xRay = xRay;
    this.scene = scene;
    this.camera = camera;
    this.renderTargetMask = new WebGLRenderTarget13(1, 1, {
      minFilter: LinearFilter10,
      magFilter: LinearFilter10,
      stencilBuffer: false,
      format: RGBFormat11
    });
    this.renderTargetMask.texture.name = "Outline.Mask";
    this.uniforms.get("maskTexture").value = this.renderTargetMask.texture;
    this.renderTargetOutline = this.renderTargetMask.clone();
    this.renderTargetOutline.texture.name = "Outline.Edges";
    this.renderTargetOutline.depthBuffer = false;
    this.renderTargetBlurredOutline = this.renderTargetOutline.clone();
    this.renderTargetBlurredOutline.texture.name = "Outline.BlurredEdges";
    this.clearPass = new ClearPass();
    this.clearPass.overrideClearColor = new Color6(0);
    this.clearPass.overrideClearAlpha = 1;
    this.depthPass = new DepthPass(scene, camera);
    this.maskPass = new RenderPass(scene, camera, new DepthComparisonMaterial(this.depthPass.texture, camera));
    const clearPass = this.maskPass.getClearPass();
    clearPass.overrideClearColor = new Color6(16777215);
    clearPass.overrideClearAlpha = 1;
    this.blurPass = new BlurPass({resolutionScale, width, height, kernelSize});
    this.blurPass.resolution.resizable = this;
    this.blur = blur;
    this.outlinePass = new ShaderPass(new OutlineMaterial());
    this.outlinePass.getFullscreenMaterial().uniforms.inputBuffer.value = this.renderTargetMask.texture;
    this.time = 0;
    this.selection = new Selection();
    this.pulseSpeed = pulseSpeed;
  }
  get resolution() {
    return this.blurPass.resolution;
  }
  get width() {
    return this.resolution.width;
  }
  set width(value) {
    this.resolution.width = value;
  }
  get height() {
    return this.resolution.height;
  }
  set height(value) {
    this.resolution.height = value;
  }
  get selectionLayer() {
    return this.selection.layer;
  }
  set selectionLayer(value) {
    this.selection.layer = value;
  }
  get dithering() {
    return this.blurPass.dithering;
  }
  set dithering(value) {
    this.blurPass.dithering = value;
  }
  get kernelSize() {
    return this.blurPass.kernelSize;
  }
  set kernelSize(value) {
    this.blurPass.kernelSize = value;
  }
  get blur() {
    return this.blurPass.enabled;
  }
  set blur(value) {
    this.blurPass.enabled = value;
    this.uniforms.get("edgeTexture").value = value ? this.renderTargetBlurredOutline.texture : this.renderTargetOutline.texture;
  }
  get xRay() {
    return this.defines.has("X_RAY");
  }
  set xRay(value) {
    if (this.xRay !== value) {
      if (value) {
        this.defines.set("X_RAY", "1");
      } else {
        this.defines.delete("X_RAY");
      }
      this.setChanged();
    }
  }
  setPatternTexture(texture) {
    if (texture !== null) {
      texture.wrapS = texture.wrapT = RepeatWrapping2;
      this.defines.set("USE_PATTERN", "1");
      this.uniforms.get("patternTexture").value = texture;
      this.setVertexShader(shader_default64);
    } else {
      this.defines.delete("USE_PATTERN");
      this.uniforms.get("patternTexture").value = null;
      this.setVertexShader(null);
    }
    this.setChanged();
  }
  getResolutionScale() {
    return this.resolution.scale;
  }
  setResolutionScale(scale) {
    this.resolution.scale = scale;
  }
  setSelection(objects) {
    this.selection.set(objects);
    return this;
  }
  clearSelection() {
    this.selection.clear();
    return this;
  }
  selectObject(object) {
    this.selection.add(object);
    return this;
  }
  deselectObject(object) {
    this.selection.delete(object);
    return this;
  }
  update(renderer, inputBuffer, deltaTime) {
    const scene = this.scene;
    const camera = this.camera;
    const selection = this.selection;
    const pulse = this.uniforms.get("pulse");
    const background = scene.background;
    const mask = camera.layers.mask;
    if (selection.size > 0) {
      scene.background = null;
      pulse.value = 1;
      if (this.pulseSpeed > 0) {
        pulse.value = 0.625 + Math.cos(this.time * this.pulseSpeed * 10) * 0.375;
      }
      this.time += deltaTime;
      selection.setVisible(false);
      this.depthPass.render(renderer);
      selection.setVisible(true);
      camera.layers.set(selection.layer);
      this.maskPass.render(renderer, this.renderTargetMask);
      camera.layers.mask = mask;
      scene.background = background;
      this.outlinePass.render(renderer, null, this.renderTargetOutline);
      if (this.blur) {
        this.blurPass.render(renderer, this.renderTargetOutline, this.renderTargetBlurredOutline);
      }
    } else if (this.time > 0) {
      this.clearPass.render(renderer, this.renderTargetMask);
      this.time = 0;
    }
  }
  setSize(width, height) {
    this.blurPass.setSize(width, height);
    this.renderTargetMask.setSize(width, height);
    const w = this.resolution.width;
    const h = this.resolution.height;
    this.depthPass.setSize(w, h);
    this.renderTargetOutline.setSize(w, h);
    this.renderTargetBlurredOutline.setSize(w, h);
    this.outlinePass.getFullscreenMaterial().setTexelSize(1 / w, 1 / h);
  }
  initialize(renderer, alpha, frameBufferType) {
    this.blurPass.initialize(renderer, alpha, UnsignedByteType14);
    if (frameBufferType !== void 0) {
      this.depthPass.initialize(renderer, alpha, frameBufferType);
      this.maskPass.initialize(renderer, alpha, frameBufferType);
      this.outlinePass.initialize(renderer, alpha, frameBufferType);
    }
  }
};

// src/effects/PixelationEffect.js
import {Uniform as Uniform34, Vector2 as Vector219} from "../build/three.module.js";

// src/effects/glsl/pixelation/shader.frag
var shader_default65 = "uniform bool active;uniform vec2 d;void mainUv(inout vec2 uv){if(active){uv=vec2(d.x*(floor(uv.x/d.x)+0.5),d.y*(floor(uv.y/d.y)+0.5));}}";

// src/effects/PixelationEffect.js
var PixelationEffect = class extends Effect {
  constructor(granularity = 30) {
    super("PixelationEffect", shader_default65, {
      uniforms: new Map([
        ["active", new Uniform34(false)],
        ["d", new Uniform34(new Vector219())]
      ])
    });
    this.resolution = new Vector219();
    this.granularity = granularity;
  }
  getGranularity() {
    return this.granularity;
  }
  setGranularity(granularity) {
    granularity = Math.floor(granularity);
    if (granularity % 2 > 0) {
      granularity += 1;
    }
    const uniforms = this.uniforms;
    uniforms.get("active").value = granularity > 0;
    uniforms.get("d").value.set(granularity, granularity).divide(this.resolution);
    this.granularity = granularity;
  }
  setSize(width, height) {
    this.resolution.set(width, height);
    this.setGranularity(this.granularity);
  }
};

// src/effects/RealisticBokehEffect.js
import {Uniform as Uniform35, Vector4 as Vector42} from "../build/three.module.js";

// src/effects/glsl/realistic-bokeh/shader.frag
var shader_default66 = "uniform float focus;uniform float focalLength;uniform float fStop;uniform float maxBlur;uniform float luminanceThreshold;uniform float luminanceGain;uniform float bias;uniform float fringe;\n#ifdef MANUAL_DOF\nuniform vec4 dof;\n#endif\n#ifdef PENTAGON\nfloat pentagon(const in vec2 coords){const vec4 HS0=vec4(1.0,0.0,0.0,1.0);const vec4 HS1=vec4(0.309016994,0.951056516,0.0,1.0);const vec4 HS2=vec4(-0.809016994,0.587785252,0.0,1.0);const vec4 HS3=vec4(-0.809016994,-0.587785252,0.0,1.0);const vec4 HS4=vec4(0.309016994,-0.951056516,0.0,1.0);const vec4 HS5=vec4(0.0,0.0,1.0,1.0);const vec4 ONE=vec4(1.0);const float P_FEATHER=0.4;const float N_FEATHER=-P_FEATHER;float inOrOut=-4.0;vec4 P=vec4(coords,vec2(RINGS_FLOAT-1.3));vec4 dist=vec4(dot(P,HS0),dot(P,HS1),dot(P,HS2),dot(P,HS3));dist=smoothstep(N_FEATHER,P_FEATHER,dist);inOrOut+=dot(dist,ONE);dist.x=dot(P,HS4);dist.y=HS5.w-abs(P.z);dist=smoothstep(N_FEATHER,P_FEATHER,dist);inOrOut+=dist.x;return clamp(inOrOut,0.0,1.0);}\n#endif\nvec3 processTexel(const in vec2 coords,const in float blur){vec2 scale=texelSize*fringe*blur;vec3 c=vec3(texture2D(inputBuffer,coords+vec2(0.0,1.0)*scale).r,texture2D(inputBuffer,coords+vec2(-0.866,-0.5)*scale).g,texture2D(inputBuffer,coords+vec2(0.866,-0.5)*scale).b);float luminance=linearToRelativeLuminance(c);float threshold=max((luminance-luminanceThreshold)*luminanceGain,0.0);return c+mix(vec3(0.0),c,threshold*blur);}float gather(const in float i,const in float j,const in float ringSamples,const in vec2 uv,const in vec2 blurFactor,const in float blur,inout vec3 color){float step=PI2/ringSamples;vec2 wh=vec2(cos(j*step)*i,sin(j*step)*i);\n#ifdef PENTAGON\nfloat p=pentagon(wh);\n#else\nfloat p=1.0;\n#endif\ncolor+=processTexel(wh*blurFactor+uv,blur)*mix(1.0,i/RINGS_FLOAT,bias)*p;return mix(1.0,i/RINGS_FLOAT,bias)*p;}void mainImage(const in vec4 inputColor,const in vec2 uv,const in float depth,out vec4 outputColor){\n#ifdef PERSPECTIVE_CAMERA\nfloat viewZ=perspectiveDepthToViewZ(depth,cameraNear,cameraFar);float linearDepth=viewZToOrthographicDepth(viewZ,cameraNear,cameraFar);\n#else\nfloat linearDepth=depth;\n#endif\n#ifdef MANUAL_DOF\nfloat focalPlane=linearDepth-focus;float farDoF=(focalPlane-dof.z)/dof.w;float nearDoF=(-focalPlane-dof.x)/dof.y;float blur=(focalPlane>0.0)? farDoF : nearDoF;\n#else\nconst float CIRCLE_OF_CONFUSION=0.03;float focalPlaneMM=focus*1000.0;float depthMM=linearDepth*1000.0;float focalPlane=(depthMM*focalLength)/(depthMM-focalLength);float farDoF=(focalPlaneMM*focalLength)/(focalPlaneMM-focalLength);float nearDoF=(focalPlaneMM-focalLength)/(focalPlaneMM*fStop*CIRCLE_OF_CONFUSION);float blur=abs(focalPlane-farDoF)*nearDoF;\n#endif\nconst int MAX_RING_SAMPLES=RINGS_INT*SAMPLES_INT;blur=clamp(blur,0.0,1.0);vec3 color=inputColor.rgb;if(blur>=0.05){vec2 blurFactor=blur*maxBlur*texelSize;float s=1.0;int ringSamples;for(int i=1;i<=RINGS_INT;i++){ringSamples=i*SAMPLES_INT;for(int j=0;j<MAX_RING_SAMPLES;j++){if(j>=ringSamples){break;}s+=gather(float(i),float(j),float(ringSamples),uv,blurFactor,blur,color);}}color/=s;}\n#ifdef SHOW_FOCUS\nfloat edge=0.002*linearDepth;float m=clamp(smoothstep(0.0,edge,blur),0.0,1.0);float e=clamp(smoothstep(1.0-edge,1.0,blur),0.0,1.0);color=mix(color,vec3(1.0,0.5,0.0),(1.0-m)*0.6);color=mix(color,vec3(0.0,0.5,1.0),((1.0-e)-(1.0-m))*0.2);\n#endif\noutputColor=vec4(color,inputColor.a);}";

// src/effects/RealisticBokehEffect.js
var RealisticBokehEffect = class extends Effect {
  constructor({
    blendFunction = BlendFunction.NORMAL,
    focus = 1,
    focalLength = 24,
    fStop = 0.9,
    luminanceThreshold = 0.5,
    luminanceGain = 2,
    bias = 0.5,
    fringe = 0.7,
    maxBlur = 1,
    rings = 3,
    samples = 2,
    showFocus = false,
    manualDoF = false,
    pentagon = false
  } = {}) {
    super("RealisticBokehEffect", shader_default66, {
      blendFunction,
      attributes: EffectAttribute.CONVOLUTION | EffectAttribute.DEPTH,
      uniforms: new Map([
        ["focus", new Uniform35(focus)],
        ["focalLength", new Uniform35(focalLength)],
        ["fStop", new Uniform35(fStop)],
        ["luminanceThreshold", new Uniform35(luminanceThreshold)],
        ["luminanceGain", new Uniform35(luminanceGain)],
        ["bias", new Uniform35(bias)],
        ["fringe", new Uniform35(fringe)],
        ["maxBlur", new Uniform35(maxBlur)],
        ["dof", new Uniform35(null)]
      ])
    });
    this.rings = rings;
    this.samples = samples;
    this.showFocus = showFocus;
    this.manualDoF = manualDoF;
    this.pentagon = pentagon;
  }
  get rings() {
    return Number.parseInt(this.defines.get("RINGS_INT"));
  }
  set rings(value) {
    const r = Math.floor(value);
    this.defines.set("RINGS_INT", r.toFixed(0));
    this.defines.set("RINGS_FLOAT", r.toFixed(1));
    this.setChanged();
  }
  get samples() {
    return Number.parseInt(this.defines.get("SAMPLES_INT"));
  }
  set samples(value) {
    const s = Math.floor(value);
    this.defines.set("SAMPLES_INT", s.toFixed(0));
    this.defines.set("SAMPLES_FLOAT", s.toFixed(1));
    this.setChanged();
  }
  get showFocus() {
    return this.defines.has("SHOW_FOCUS");
  }
  set showFocus(value) {
    if (this.showFocus !== value) {
      if (value) {
        this.defines.set("SHOW_FOCUS", "1");
      } else {
        this.defines.delete("SHOW_FOCUS");
      }
      this.setChanged();
    }
  }
  get manualDoF() {
    return this.defines.has("MANUAL_DOF");
  }
  set manualDoF(value) {
    if (this.manualDoF !== value) {
      if (value) {
        this.defines.set("MANUAL_DOF", "1");
        this.uniforms.get("dof").value = new Vector42(0.2, 1, 0.2, 2);
      } else {
        this.defines.delete("MANUAL_DOF");
        this.uniforms.get("dof").value = null;
      }
      this.setChanged();
    }
  }
  get pentagon() {
    return this.defines.has("PENTAGON");
  }
  set pentagon(value) {
    if (this.pentagon !== value) {
      if (value) {
        this.defines.set("PENTAGON", "1");
      } else {
        this.defines.delete("PENTAGON");
      }
      this.setChanged();
    }
  }
};

// src/effects/ScanlineEffect.js
import {Uniform as Uniform36, Vector2 as Vector220} from "../build/three.module.js";

// src/effects/glsl/scanlines/shader.frag
var shader_default67 = "uniform float count;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec2 sl=vec2(sin(uv.y*count),cos(uv.y*count));vec3 scanlines=vec3(sl.x,sl.y,sl.x);outputColor=vec4(scanlines,inputColor.a);}";

// src/effects/ScanlineEffect.js
var ScanlineEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.OVERLAY, density = 1.25} = {}) {
    super("ScanlineEffect", shader_default67, {
      blendFunction,
      uniforms: new Map([
        ["count", new Uniform36(0)]
      ])
    });
    this.resolution = new Vector220();
    this.density = density;
  }
  getDensity() {
    return this.density;
  }
  setDensity(density) {
    this.density = density;
    this.setSize(this.resolution.x, this.resolution.y);
  }
  setSize(width, height) {
    this.resolution.set(width, height);
    this.uniforms.get("count").value = Math.round(height * this.density);
  }
};

// src/effects/ShockWaveEffect.js
import {Uniform as Uniform37, Vector2 as Vector221, Vector3 as Vector35} from "../build/three.module.js";

// src/effects/glsl/shock-wave/shader.frag
var shader_default68 = "uniform bool active;uniform vec2 center;uniform float waveSize;uniform float radius;uniform float maxRadius;uniform float amplitude;varying float vSize;void mainUv(inout vec2 uv){if(active){vec2 aspectCorrection=vec2(aspect,1.0);vec2 difference=uv*aspectCorrection-center*aspectCorrection;float distance=sqrt(dot(difference,difference))*vSize;if(distance>radius){if(distance<radius+waveSize){float angle=(distance-radius)*PI2/waveSize;float cosSin=(1.0-cos(angle))*0.5;float extent=maxRadius+waveSize;float decay=max(extent-distance*distance,0.0)/extent;uv-=((cosSin*amplitude*difference)/distance)*decay;}}}}";

// src/effects/glsl/shock-wave/shader.vert
var shader_default69 = "uniform float size;uniform float cameraDistance;varying float vSize;void mainSupport(){vSize=(0.1*cameraDistance)/size;}";

// src/effects/ShockWaveEffect.js
var HALF_PI = Math.PI * 0.5;
var v2 = new Vector35();
var ab = new Vector35();
var ShockWaveEffect = class extends Effect {
  constructor(camera, epicenter = new Vector35(), {
    speed = 2,
    maxRadius = 1,
    waveSize = 0.2,
    amplitude = 0.05
  } = {}) {
    super("ShockWaveEffect", shader_default68, {
      vertexShader: shader_default69,
      uniforms: new Map([
        ["active", new Uniform37(false)],
        ["center", new Uniform37(new Vector221(0.5, 0.5))],
        ["cameraDistance", new Uniform37(1)],
        ["size", new Uniform37(1)],
        ["radius", new Uniform37(-waveSize)],
        ["maxRadius", new Uniform37(maxRadius)],
        ["waveSize", new Uniform37(waveSize)],
        ["amplitude", new Uniform37(amplitude)]
      ])
    });
    this.camera = camera;
    this.epicenter = epicenter;
    this.screenPosition = this.uniforms.get("center").value;
    this.speed = speed;
    this.time = 0;
    this.active = false;
  }
  explode() {
    this.time = 0;
    this.active = true;
    this.uniforms.get("active").value = true;
  }
  update(renderer, inputBuffer, delta) {
    const epicenter = this.epicenter;
    const camera = this.camera;
    const uniforms = this.uniforms;
    const uniformActive = uniforms.get("active");
    if (this.active) {
      const waveSize = uniforms.get("waveSize").value;
      camera.getWorldDirection(v2);
      ab.copy(camera.position).sub(epicenter);
      uniformActive.value = v2.angleTo(ab) > HALF_PI;
      if (uniformActive.value) {
        uniforms.get("cameraDistance").value = camera.position.distanceTo(epicenter);
        v2.copy(epicenter).project(camera);
        this.screenPosition.set((v2.x + 1) * 0.5, (v2.y + 1) * 0.5);
      }
      this.time += delta * this.speed;
      const radius = this.time - waveSize;
      uniforms.get("radius").value = radius;
      if (radius >= (uniforms.get("maxRadius").value + waveSize) * 2) {
        this.active = false;
        uniformActive.value = false;
      }
    }
  }
};

// src/effects/SelectiveBloomEffect.js
import {
  BasicDepthPacking as BasicDepthPacking6,
  Color as Color7,
  LinearFilter as LinearFilter11,
  NotEqualDepth as NotEqualDepth2,
  EqualDepth as EqualDepth2,
  RGBADepthPacking as RGBADepthPacking4,
  RGBFormat as RGBFormat12,
  UnsignedByteType as UnsignedByteType15,
  WebGLRenderTarget as WebGLRenderTarget14
} from "../build/three.module.js";
var SelectiveBloomEffect = class extends BloomEffect {
  constructor(scene, camera, options) {
    super(options);
    this.setAttributes(this.getAttributes() | EffectAttribute.DEPTH);
    this.camera = camera;
    this.depthPass = new DepthPass(scene, camera);
    this.clearPass = new ClearPass(true, false, false);
    this.clearPass.overrideClearColor = new Color7(0);
    this.depthMaskPass = new ShaderPass(new DepthMaskMaterial());
    const depthMaskMaterial = this.depthMaskMaterial;
    depthMaskMaterial.uniforms.depthBuffer1.value = this.depthPass.texture;
    depthMaskMaterial.defines.DEPTH_PACKING_1 = RGBADepthPacking4.toFixed(0);
    depthMaskMaterial.setDepthMode(EqualDepth2);
    this.renderTargetMasked = new WebGLRenderTarget14(1, 1, {
      minFilter: LinearFilter11,
      magFilter: LinearFilter11,
      stencilBuffer: false,
      depthBuffer: false
    });
    this.renderTargetMasked.texture.name = "Bloom.Masked";
    this.renderTargetMasked.texture.generateMipmaps = false;
    this.selection = new Selection();
  }
  get depthMaskMaterial() {
    return this.depthMaskPass.getFullscreenMaterial();
  }
  get inverted() {
    return this.depthMaskMaterial.getDepthMode() === NotEqualDepth2;
  }
  set inverted(value) {
    this.depthMaskMaterial.setDepthMode(value ? NotEqualDepth2 : EqualDepth2);
  }
  get ignoreBackground() {
    return !this.depthMaskMaterial.keepFar;
  }
  set ignoreBackground(value) {
    this.depthMaskMaterial.keepFar = !value;
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking6) {
    const material = this.depthMaskPass.getFullscreenMaterial();
    material.uniforms.depthBuffer0.value = depthTexture;
    material.defines.DEPTH_PACKING_0 = depthPacking.toFixed(0);
    material.needsUpdate = true;
  }
  update(renderer, inputBuffer, deltaTime) {
    const camera = this.camera;
    const selection = this.selection;
    const renderTarget = this.renderTargetMasked;
    const mask = camera.layers.mask;
    camera.layers.set(selection.layer);
    this.depthPass.render(renderer);
    camera.layers.mask = mask;
    this.clearPass.render(renderer, renderTarget);
    this.depthMaskPass.render(renderer, inputBuffer, renderTarget);
    super.update(renderer, renderTarget, deltaTime);
  }
  setSize(width, height) {
    super.setSize(width, height);
    this.clearPass.setSize(width, height);
    this.depthPass.setSize(width, height);
    this.depthMaskPass.setSize(width, height);
    this.renderTargetMasked.setSize(width, height);
  }
  initialize(renderer, alpha, frameBufferType) {
    super.initialize(renderer, alpha, frameBufferType);
    this.clearPass.initialize(renderer, alpha, frameBufferType);
    this.depthPass.initialize(renderer, alpha, frameBufferType);
    this.depthMaskPass.initialize(renderer, alpha, frameBufferType);
    if (!alpha && frameBufferType === UnsignedByteType15) {
      this.renderTargetMasked.texture.format = RGBFormat12;
    }
    if (frameBufferType !== void 0) {
      this.renderTargetMasked.texture.type = frameBufferType;
    }
  }
};

// src/effects/SepiaEffect.js
import {Uniform as Uniform38} from "../build/three.module.js";

// src/effects/glsl/sepia/shader.frag
var shader_default70 = "uniform float intensity;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec3 color=vec3(dot(inputColor.rgb,vec3(1.0-0.607*intensity,0.769*intensity,0.189*intensity)),dot(inputColor.rgb,vec3(0.349*intensity,1.0-0.314*intensity,0.168*intensity)),dot(inputColor.rgb,vec3(0.272*intensity,0.534*intensity,1.0-0.869*intensity)));outputColor=vec4(color,inputColor.a);}";

// src/effects/SepiaEffect.js
var SepiaEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, intensity = 1} = {}) {
    super("SepiaEffect", shader_default70, {
      blendFunction,
      uniforms: new Map([
        ["intensity", new Uniform38(intensity)]
      ])
    });
  }
};

// src/effects/SMAAEffect.js
import {
  BasicDepthPacking as BasicDepthPacking7,
  Color as Color8,
  LinearFilter as LinearFilter12,
  NearestFilter as NearestFilter8,
  RGBAFormat as RGBAFormat6,
  RGBFormat as RGBFormat13,
  Texture,
  Uniform as Uniform39,
  Vector2 as Vector222,
  WebGLRenderTarget as WebGLRenderTarget15
} from "../build/three.module.js";

// src/images/smaa/searchImageDataURL.js
var searchImageDataURL_default = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAQCAYAAACm53kpAAAAeElEQVRYR+2XSwqAMAxEJ168ePEqwRSKhIIiuHjJqiU0gWE+1CQdApcVAMUAuARaMGCX1MIL/Ow13++9lW2s3mW9MWvsnWc/2fvGygwPAN4E8QzAA4CXAB6AHjG4JTHYI1ey3pcx6FHnEfhLDOIBKAmUBK6/ANUDTlROXAHd9EC1AAAAAElFTkSuQmCC";

// src/images/smaa/areaImageDataURL.js
var areaImageDataURL_default = "data:image/png;base64,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";

// src/effects/glsl/smaa/shader.frag
var shader_default71 = "uniform sampler2D weightMap;varying vec2 vOffset0;varying vec2 vOffset1;void movec(const in bvec2 c,inout vec2 variable,const in vec2 value){if(c.x){variable.x=value.x;}if(c.y){variable.y=value.y;}}void movec(const in bvec4 c,inout vec4 variable,const in vec4 value){movec(c.xy,variable.xy,value.xy);movec(c.zw,variable.zw,value.zw);}void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){vec4 a;a.x=texture2D(weightMap,vOffset0).a;a.y=texture2D(weightMap,vOffset1).g;a.wz=texture2D(weightMap,uv).rb;vec4 color=inputColor;if(dot(a,vec4(1.0))>=1e-5){bool h=max(a.x,a.z)>max(a.y,a.w);vec4 blendingOffset=vec4(0.0,a.y,0.0,a.w);vec2 blendingWeight=a.yw;movec(bvec4(h),blendingOffset,vec4(a.x,0.0,a.z,0.0));movec(bvec2(h),blendingWeight,a.xz);blendingWeight/=dot(blendingWeight,vec2(1.0));vec4 blendingCoord=blendingOffset*vec4(texelSize,-texelSize)+uv.xyxy;color=blendingWeight.x*texture2D(inputBuffer,blendingCoord.xy);color+=blendingWeight.y*texture2D(inputBuffer,blendingCoord.zw);}outputColor=color;}";

// src/effects/glsl/smaa/shader.vert
var shader_default72 = "varying vec2 vOffset0;varying vec2 vOffset1;void mainSupport(const in vec2 uv){vOffset0=uv+texelSize*vec2(1.0,0.0);vOffset1=uv+texelSize*vec2(0.0,1.0);}";

// src/effects/SMAAEffect.js
var SMAAEffect = class extends Effect {
  constructor(searchImage, areaImage, preset = SMAAPreset.HIGH, edgeDetectionMode = EdgeDetectionMode.COLOR) {
    super("SMAAEffect", shader_default71, {
      vertexShader: shader_default72,
      blendFunction: BlendFunction.NORMAL,
      attributes: EffectAttribute.CONVOLUTION | EffectAttribute.DEPTH,
      uniforms: new Map([
        ["weightMap", new Uniform39(null)]
      ])
    });
    this.renderTargetEdges = new WebGLRenderTarget15(1, 1, {
      minFilter: LinearFilter12,
      stencilBuffer: false,
      depthBuffer: false,
      format: RGBFormat13
    });
    this.renderTargetEdges.texture.name = "SMAA.Edges";
    this.renderTargetWeights = this.renderTargetEdges.clone();
    this.renderTargetWeights.texture.name = "SMAA.Weights";
    this.renderTargetWeights.texture.format = RGBAFormat6;
    this.uniforms.get("weightMap").value = this.renderTargetWeights.texture;
    this.clearPass = new ClearPass(true, false, false);
    this.clearPass.overrideClearColor = new Color8(0);
    this.clearPass.overrideClearAlpha = 1;
    this.edgeDetectionPass = new ShaderPass(new EdgeDetectionMaterial(new Vector222(), edgeDetectionMode));
    this.weightsPass = new ShaderPass(new SMAAWeightsMaterial());
    const searchTexture = new Texture(searchImage);
    searchTexture.name = "SMAA.Search";
    searchTexture.magFilter = NearestFilter8;
    searchTexture.minFilter = NearestFilter8;
    searchTexture.format = RGBAFormat6;
    searchTexture.generateMipmaps = false;
    searchTexture.needsUpdate = true;
    searchTexture.flipY = true;
    const areaTexture = new Texture(areaImage);
    areaTexture.name = "SMAA.Area";
    areaTexture.magFilter = LinearFilter12;
    areaTexture.minFilter = LinearFilter12;
    areaTexture.format = RGBAFormat6;
    areaTexture.generateMipmaps = false;
    areaTexture.needsUpdate = true;
    areaTexture.flipY = false;
    const weightsMaterial = this.weightsPass.getFullscreenMaterial();
    weightsMaterial.uniforms.searchTexture.value = searchTexture;
    weightsMaterial.uniforms.areaTexture.value = areaTexture;
    this.applyPreset(preset);
  }
  get edgeDetectionMaterial() {
    return this.edgeDetectionPass.getFullscreenMaterial();
  }
  get colorEdgesMaterial() {
    return this.edgeDetectionMaterial;
  }
  get weightsMaterial() {
    return this.weightsPass.getFullscreenMaterial();
  }
  setEdgeDetectionThreshold(threshold) {
    this.edgeDetectionPass.getFullscreenMaterial().setEdgeDetectionThreshold(threshold);
  }
  setOrthogonalSearchSteps(steps) {
    this.weightsPass.getFullscreenMaterial().setOrthogonalSearchSteps(steps);
  }
  applyPreset(preset) {
    const edgeDetectionMaterial = this.edgeDetectionMaterial;
    const weightsMaterial = this.weightsMaterial;
    switch (preset) {
      case SMAAPreset.LOW:
        edgeDetectionMaterial.setEdgeDetectionThreshold(0.15);
        weightsMaterial.setOrthogonalSearchSteps(4);
        weightsMaterial.diagonalDetection = false;
        weightsMaterial.cornerRounding = false;
        break;
      case SMAAPreset.MEDIUM:
        edgeDetectionMaterial.setEdgeDetectionThreshold(0.1);
        weightsMaterial.setOrthogonalSearchSteps(8);
        weightsMaterial.diagonalDetection = false;
        weightsMaterial.cornerRounding = false;
        break;
      case SMAAPreset.HIGH:
        edgeDetectionMaterial.setEdgeDetectionThreshold(0.1);
        weightsMaterial.setOrthogonalSearchSteps(16);
        weightsMaterial.setDiagonalSearchSteps(8);
        weightsMaterial.setCornerRounding(25);
        weightsMaterial.diagonalDetection = true;
        weightsMaterial.cornerRounding = true;
        break;
      case SMAAPreset.ULTRA:
        edgeDetectionMaterial.setEdgeDetectionThreshold(0.05);
        weightsMaterial.setOrthogonalSearchSteps(32);
        weightsMaterial.setDiagonalSearchSteps(16);
        weightsMaterial.setCornerRounding(25);
        weightsMaterial.diagonalDetection = true;
        weightsMaterial.cornerRounding = true;
        break;
    }
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking7) {
    const material = this.edgeDetectionMaterial;
    material.uniforms.depthBuffer.value = depthTexture;
    material.depthPacking = depthPacking;
  }
  update(renderer, inputBuffer, deltaTime) {
    this.clearPass.render(renderer, this.renderTargetEdges);
    this.edgeDetectionPass.render(renderer, inputBuffer, this.renderTargetEdges);
    this.weightsPass.render(renderer, this.renderTargetEdges, this.renderTargetWeights);
  }
  setSize(width, height) {
    const edgeDetectionMaterial = this.edgeDetectionPass.getFullscreenMaterial();
    const weightsMaterial = this.weightsPass.getFullscreenMaterial();
    this.renderTargetEdges.setSize(width, height);
    this.renderTargetWeights.setSize(width, height);
    weightsMaterial.uniforms.resolution.value.set(width, height);
    weightsMaterial.uniforms.texelSize.value.set(1 / width, 1 / height);
    edgeDetectionMaterial.uniforms.texelSize.value.copy(weightsMaterial.uniforms.texelSize.value);
  }
  dispose() {
    const uniforms = this.weightsPass.getFullscreenMaterial().uniforms;
    uniforms.searchTexture.value.dispose();
    uniforms.areaTexture.value.dispose();
    super.dispose();
  }
  static get searchImageDataURL() {
    return searchImageDataURL_default;
  }
  static get areaImageDataURL() {
    return areaImageDataURL_default;
  }
};
var SMAAPreset = {
  LOW: 0,
  MEDIUM: 1,
  HIGH: 2,
  ULTRA: 3
};

// src/effects/SSAOEffect.js
import {
  BasicDepthPacking as BasicDepthPacking8,
  Color as Color9,
  LinearFilter as LinearFilter13,
  RepeatWrapping as RepeatWrapping3,
  RGBFormat as RGBFormat14,
  Uniform as Uniform40,
  WebGLRenderTarget as WebGLRenderTarget16
} from "../build/three.module.js";

// src/effects/glsl/ssao/shader.frag
var shader_default73 = "uniform lowp sampler2D aoBuffer;uniform float luminanceInfluence;\n#ifdef DEPTH_AWARE_UPSAMPLING\n#ifdef GL_FRAGMENT_PRECISION_HIGH\nuniform highp sampler2D normalDepthBuffer;\n#else\nuniform mediump sampler2D normalDepthBuffer;\n#endif\n#endif\n#ifdef COLORIZE\nuniform vec3 color;\n#endif\nvoid mainImage(const in vec4 inputColor,const in vec2 uv,const in float depth,out vec4 outputColor){float aoLinear=texture2D(aoBuffer,uv).r;\n#if defined(DEPTH_AWARE_UPSAMPLING) && __VERSION__ == 300\nvec4 normalDepth[4];normalDepth[0]=textureOffset(normalDepthBuffer,uv,ivec2(0,0));normalDepth[1]=textureOffset(normalDepthBuffer,uv,ivec2(0,1));normalDepth[2]=textureOffset(normalDepthBuffer,uv,ivec2(1,0));normalDepth[3]=textureOffset(normalDepthBuffer,uv,ivec2(1,1));float dot01=dot(normalDepth[0].rgb,normalDepth[1].rgb);float dot02=dot(normalDepth[0].rgb,normalDepth[2].rgb);float dot03=dot(normalDepth[0].rgb,normalDepth[3].rgb);float minDot=min(dot01,min(dot02,dot03));float s=step(THRESHOLD,minDot);float smallestDistance=1.0;int index;for(int i=0;i<4;++i){float distance=abs(depth-normalDepth[i].a);if(distance<smallestDistance){smallestDistance=distance;index=i;}}ivec2 offsets[4];offsets[0]=ivec2(0,0);offsets[1]=ivec2(0,1);offsets[2]=ivec2(1,0);offsets[3]=ivec2(1,1);ivec2 coord=ivec2(uv*vec2(textureSize(aoBuffer,0)))+offsets[index];float aoNearest=texelFetch(aoBuffer,coord,0).r;float ao=mix(aoNearest,aoLinear,s);\n#else\nfloat ao=aoLinear;\n#endif\nfloat l=linearToRelativeLuminance(inputColor.rgb);ao=mix(ao,1.0,l*luminanceInfluence);\n#ifdef COLORIZE\noutputColor=vec4(1.0-(1.0-ao)*(1.0-color),inputColor.a);\n#else\noutputColor=vec4(vec3(ao),inputColor.a);\n#endif\n}";

// src/effects/SSAOEffect.js
var NOISE_TEXTURE_SIZE = 64;
var SSAOEffect = class extends Effect {
  constructor(camera, normalBuffer, {
    blendFunction = BlendFunction.MULTIPLY,
    distanceScaling = true,
    depthAwareUpsampling = true,
    normalDepthBuffer = null,
    samples = 9,
    rings = 7,
    distanceThreshold = 0.97,
    distanceFalloff = 0.03,
    rangeThreshold = 5e-4,
    rangeFalloff = 1e-3,
    minRadiusScale = 0.33,
    luminanceInfluence = 0.7,
    radius = 0.1825,
    intensity = 1,
    bias = 0.025,
    fade = 0.01,
    color: color2 = null,
    resolutionScale = 1,
    width = Resizer.AUTO_SIZE,
    height = Resizer.AUTO_SIZE
  } = {}) {
    super("SSAOEffect", shader_default73, {
      blendFunction,
      attributes: EffectAttribute.DEPTH,
      defines: new Map([
        ["THRESHOLD", "0.997"]
      ]),
      uniforms: new Map([
        ["aoBuffer", new Uniform40(null)],
        ["normalDepthBuffer", new Uniform40(null)],
        ["luminanceInfluence", new Uniform40(luminanceInfluence)],
        ["color", new Uniform40(null)],
        ["scale", new Uniform40(0)]
      ])
    });
    this.renderTargetAO = new WebGLRenderTarget16(1, 1, {
      minFilter: LinearFilter13,
      magFilter: LinearFilter13,
      stencilBuffer: false,
      depthBuffer: false,
      format: RGBFormat14
    });
    this.renderTargetAO.texture.name = "AO.Target";
    this.renderTargetAO.texture.generateMipmaps = false;
    this.uniforms.get("aoBuffer").value = this.renderTargetAO.texture;
    this.resolution = new Resizer(this, width, height, resolutionScale);
    this.r = 1;
    this.camera = camera;
    this.ssaoPass = new ShaderPass((() => {
      const noiseTexture = new NoiseTexture(NOISE_TEXTURE_SIZE, NOISE_TEXTURE_SIZE);
      noiseTexture.wrapS = noiseTexture.wrapT = RepeatWrapping3;
      const material = new SSAOMaterial(camera);
      material.uniforms.noiseTexture.value = noiseTexture;
      material.uniforms.intensity.value = intensity;
      material.uniforms.minRadiusScale.value = minRadiusScale;
      material.uniforms.fade.value = fade;
      material.uniforms.bias.value = bias;
      if (normalDepthBuffer !== null) {
        material.uniforms.normalDepthBuffer.value = normalDepthBuffer;
        material.defines.NORMAL_DEPTH = "1";
        if (depthAwareUpsampling) {
          this.depthAwareUpsampling = depthAwareUpsampling;
          this.uniforms.get("normalDepthBuffer").value = normalDepthBuffer;
        }
      } else {
        material.uniforms.normalBuffer.value = normalBuffer;
      }
      return material;
    })());
    this.distanceScaling = distanceScaling;
    this.samples = samples;
    this.rings = rings;
    this.color = color2;
    this.radius = radius > 1 ? radius / 100 : radius;
    this.setDistanceCutoff(distanceThreshold, distanceFalloff);
    this.setProximityCutoff(rangeThreshold, rangeFalloff);
  }
  get ssaoMaterial() {
    return this.ssaoPass.getFullscreenMaterial();
  }
  get samples() {
    return Number(this.ssaoMaterial.defines.SAMPLES_INT);
  }
  set samples(value) {
    const material = this.ssaoMaterial;
    material.defines.SAMPLES_INT = value.toFixed(0);
    material.defines.SAMPLES_FLOAT = value.toFixed(1);
    material.needsUpdate = true;
  }
  get rings() {
    return Number(this.ssaoMaterial.defines.SPIRAL_TURNS);
  }
  set rings(value) {
    const material = this.ssaoMaterial;
    material.defines.SPIRAL_TURNS = value.toFixed(1);
    material.needsUpdate = true;
  }
  get radius() {
    return this.r;
  }
  set radius(value) {
    this.r = Math.min(Math.max(value, 1e-6), 1);
    const radius = this.r * this.resolution.height;
    const material = this.ssaoMaterial;
    material.defines.RADIUS = radius.toFixed(11);
    material.defines.RADIUS_SQ = (radius * radius).toFixed(11);
    material.needsUpdate = true;
  }
  get depthAwareUpsampling() {
    return this.defines.has("DEPTH_AWARE_UPSAMPLING");
  }
  set depthAwareUpsampling(value) {
    if (this.depthAwareUpsampling !== value) {
      if (value) {
        this.defines.set("DEPTH_AWARE_UPSAMPLING", "1");
      } else {
        this.defines.delete("DEPTH_AWARE_UPSAMPLING");
      }
      this.setChanged();
    }
  }
  get distanceScaling() {
    return this.ssaoMaterial.defines.DISTANCE_SCALING !== void 0;
  }
  set distanceScaling(value) {
    if (this.distanceScaling !== value) {
      const material = this.ssaoMaterial;
      if (value) {
        material.defines.DISTANCE_SCALING = "1";
      } else {
        delete material.defines.DISTANCE_SCALING;
      }
      material.needsUpdate = true;
    }
  }
  get color() {
    return this.uniforms.get("color").value;
  }
  set color(value) {
    const uniforms = this.uniforms;
    const defines = this.defines;
    if (value !== null) {
      if (defines.has("COLORIZE")) {
        uniforms.get("color").value.set(value);
      } else {
        defines.set("COLORIZE", "1");
        uniforms.get("color").value = new Color9(value);
        this.setChanged();
      }
    } else if (defines.has("COLORIZE")) {
      defines.delete("COLORIZE");
      uniforms.get("color").value = null;
      this.setChanged();
    }
  }
  setDistanceCutoff(threshold, falloff) {
    this.ssaoMaterial.uniforms.distanceCutoff.value.set(Math.min(Math.max(threshold, 0), 1), Math.min(Math.max(threshold + falloff, 0), 1));
  }
  setProximityCutoff(threshold, falloff) {
    this.ssaoMaterial.uniforms.proximityCutoff.value.set(Math.min(Math.max(threshold, 0), 1), Math.min(Math.max(threshold + falloff, 0), 1));
  }
  setDepthTexture(depthTexture, depthPacking = BasicDepthPacking8) {
    const material = this.ssaoMaterial;
    if (material.defines.NORMAL_DEPTH === void 0) {
      material.uniforms.normalDepthBuffer.value = depthTexture;
      material.depthPacking = depthPacking;
    }
  }
  update(renderer, inputBuffer, deltaTime) {
    this.ssaoPass.render(renderer, null, this.renderTargetAO);
  }
  setSize(width, height) {
    const resolution = this.resolution;
    resolution.base.set(width, height);
    const w = resolution.width;
    const h = resolution.height;
    this.renderTargetAO.setSize(w, h);
    this.ssaoMaterial.setTexelSize(1 / w, 1 / h);
    const camera = this.camera;
    const uniforms = this.ssaoMaterial.uniforms;
    uniforms.noiseScale.value.set(w, h).divideScalar(NOISE_TEXTURE_SIZE);
    uniforms.inverseProjectionMatrix.value.copy(camera.projectionMatrix).invert();
    uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
    this.radius = this.r;
  }
};

// src/effects/TextureEffect.js
import {
  LinearEncoding as LinearEncoding3,
  Matrix3,
  sRGBEncoding as sRGBEncoding3,
  Uniform as Uniform41,
  UnsignedByteType as UnsignedByteType16
} from "../build/three.module.js";

// src/effects/glsl/texture/shader.frag
var shader_default74 = "#ifdef TEXTURE_PRECISION_HIGH\nuniform mediump sampler2D map;\n#else\nuniform lowp sampler2D map;\n#endif\n#if defined(ASPECT_CORRECTION) || defined(UV_TRANSFORM)\nvarying vec2 vUv2;\n#endif\nvoid mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){\n#if defined(ASPECT_CORRECTION) || defined(UV_TRANSFORM)\nvec4 texel=texelToLinear(texture2D(map,vUv2));\n#else\nvec4 texel=texelToLinear(texture2D(map,uv));\n#endif\noutputColor=TEXEL;}";

// src/effects/glsl/texture/shader.vert
var shader_default75 = "#ifdef ASPECT_CORRECTION\nuniform float scale;\n#else\nuniform mat3 uvTransform;\n#endif\nvarying vec2 vUv2;void mainSupport(const in vec2 uv){\n#ifdef ASPECT_CORRECTION\nvUv2=uv*vec2(aspect,1.0)*scale;\n#else\nvUv2=(uvTransform*vec3(uv,1.0)).xy;\n#endif\n}";

// src/effects/TextureEffect.js
var TextureEffect = class extends Effect {
  constructor({blendFunction = BlendFunction.NORMAL, texture = null, aspectCorrection = false} = {}) {
    super("TextureEffect", shader_default74, {
      blendFunction,
      defines: new Map([
        ["TEXEL", "texel"]
      ]),
      uniforms: new Map([
        ["map", new Uniform41(null)],
        ["scale", new Uniform41(1)],
        ["uvTransform", new Uniform41(null)]
      ])
    });
    this.texture = texture;
    this.aspectCorrection = aspectCorrection;
  }
  get texture() {
    return this.uniforms.get("map").value;
  }
  set texture(value) {
    const currentTexture = this.texture;
    if (currentTexture !== value) {
      this.uniforms.get("map").value = value;
      this.defines.delete("TEXTURE_PRECISION_HIGH");
      if (value !== null) {
        switch (value.encoding) {
          case sRGBEncoding3:
            this.defines.set("texelToLinear(texel)", "sRGBToLinear(texel)");
            break;
          case LinearEncoding3:
            this.defines.set("texelToLinear(texel)", "texel");
            break;
          default:
            console.error("Unsupported encoding:", value.encoding);
            break;
        }
        if (value.type !== UnsignedByteType16) {
          this.defines.set("TEXTURE_PRECISION_HIGH", "1");
        }
        if (currentTexture === null || currentTexture.type !== value.type || currentTexture.encoding !== value.encoding) {
          this.setChanged();
        }
      }
    }
  }
  get aspectCorrection() {
    return this.defines.has("ASPECT_CORRECTION");
  }
  set aspectCorrection(value) {
    if (this.aspectCorrection !== value) {
      if (value) {
        if (this.uvTransform) {
          this.uvTransform = false;
        }
        this.defines.set("ASPECT_CORRECTION", "1");
        this.setVertexShader(shader_default75);
      } else {
        this.defines.delete("ASPECT_CORRECTION");
        this.setVertexShader(null);
      }
      this.setChanged();
    }
  }
  get uvTransform() {
    return this.defines.has("UV_TRANSFORM");
  }
  set uvTransform(value) {
    if (this.uvTransform !== value) {
      if (value) {
        if (this.aspectCorrection) {
          this.aspectCorrection = false;
        }
        this.defines.set("UV_TRANSFORM", "1");
        this.uniforms.get("uvTransform").value = new Matrix3();
        this.setVertexShader(shader_default75);
      } else {
        this.defines.delete("UV_TRANSFORM");
        this.uniforms.get("uvTransform").value = null;
        this.setVertexShader(null);
      }
      this.setChanged();
    }
  }
  setTextureSwizzleRGBA(r, g = r, b = r, a = r) {
    const rgba = "rgba";
    let swizzle = "";
    if (r !== ColorChannel.RED || g !== ColorChannel.GREEN || b !== ColorChannel.BLUE || a !== ColorChannel.ALPHA) {
      swizzle = [".", rgba[r], rgba[g], rgba[b], rgba[a]].join("");
    }
    this.defines.set("TEXEL", "texel" + swizzle);
    this.setChanged();
  }
  update(renderer, inputBuffer, deltaTime) {
    const texture = this.uniforms.get("map").value;
    if (this.uvTransform && texture.matrixAutoUpdate) {
      texture.updateMatrix();
      this.uniforms.get("uvTransform").value.copy(texture.matrix);
    }
  }
};

// src/effects/ToneMappingEffect.js
import {
  LinearFilter as LinearFilter14,
  LinearMipMapLinearFilter,
  LinearMipmapLinearFilter,
  RGBFormat as RGBFormat15,
  Uniform as Uniform42,
  WebGLRenderTarget as WebGLRenderTarget17
} from "../build/three.module.js";

// src/effects/glsl/tone-mapping/shader.frag
var shader_default76 = "#include <tonemapping_pars_fragment>\nuniform lowp sampler2D luminanceBuffer;uniform float whitePoint;uniform float middleGrey;\n#ifndef ADAPTIVE\nuniform float averageLuminance;\n#endif\nvec3 Reinhard2ToneMapping(vec3 color){color*=toneMappingExposure;float l=linearToRelativeLuminance(color);\n#ifdef ADAPTIVE\nfloat lumAvg=texture2D(luminanceBuffer,vec2(0.5)).r;\n#else\nfloat lumAvg=averageLuminance;\n#endif\nfloat lumScaled=(l*middleGrey)/max(lumAvg,1e-6);float lumCompressed=lumScaled*(1.0+lumScaled/(whitePoint*whitePoint));lumCompressed/=(1.0+lumScaled);return clamp(lumCompressed*color,0.0,1.0);}void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){\n#ifdef REINHARD2\noutputColor=vec4(Reinhard2ToneMapping(inputColor.rgb),inputColor.a);\n#else\noutputColor=vec4(toneMapping(inputColor.rgb),inputColor.a);\n#endif\n}";

// src/effects/ToneMappingEffect.js
var ToneMappingEffect = class extends Effect {
  constructor({
    blendFunction = BlendFunction.NORMAL,
    adaptive = true,
    mode = adaptive ? ToneMappingMode.REINHARD2_ADAPTIVE : ToneMappingMode.REINHARD2,
    resolution = 256,
    maxLuminance = 16,
    whitePoint = maxLuminance,
    middleGrey = 0.6,
    minLuminance = 0.01,
    averageLuminance = 1,
    adaptationRate = 1
  } = {}) {
    super("ToneMappingEffect", shader_default76, {
      blendFunction,
      uniforms: new Map([
        ["luminanceBuffer", new Uniform42(null)],
        ["maxLuminance", new Uniform42(maxLuminance)],
        ["whitePoint", new Uniform42(whitePoint)],
        ["middleGrey", new Uniform42(middleGrey)],
        ["averageLuminance", new Uniform42(averageLuminance)]
      ])
    });
    this.renderTargetLuminance = new WebGLRenderTarget17(1, 1, {
      minFilter: LinearMipmapLinearFilter !== void 0 ? LinearMipmapLinearFilter : LinearMipMapLinearFilter,
      magFilter: LinearFilter14,
      stencilBuffer: false,
      depthBuffer: false,
      format: RGBFormat15
    });
    this.renderTargetLuminance.texture.name = "Luminance";
    this.renderTargetLuminance.texture.generateMipmaps = true;
    this.luminancePass = new LuminancePass({
      renderTarget: this.renderTargetLuminance
    });
    this.adaptiveLuminancePass = new AdaptiveLuminancePass(this.luminancePass.texture, {
      minLuminance,
      adaptationRate
    });
    this.uniforms.get("luminanceBuffer").value = this.adaptiveLuminancePass.texture;
    this.mode = null;
    this.setMode(mode);
    this.resolution = resolution;
  }
  getMode() {
    return this.mode;
  }
  setMode(value) {
    const currentMode = this.mode;
    if (currentMode !== value) {
      this.defines.clear();
      switch (value) {
        case ToneMappingMode.REINHARD:
          this.defines.set("toneMapping(texel)", "ReinhardToneMapping(texel)");
          break;
        case ToneMappingMode.OPTIMIZED_CINEON:
          this.defines.set("toneMapping(texel)", "OptimizedCineonToneMapping(texel)");
          break;
        case ToneMappingMode.ACES_FILMIC:
          this.defines.set("toneMapping(texel)", "ACESFilmicToneMapping(texel)");
          break;
        default:
          this.defines.set("toneMapping(texel)", "texel");
          break;
      }
      if (value === ToneMappingMode.REINHARD2) {
        this.defines.set("REINHARD2", "1");
      } else if (value === ToneMappingMode.REINHARD2_ADAPTIVE) {
        this.defines.set("REINHARD2", "1");
        this.defines.set("ADAPTIVE", "1");
      }
      this.mode = value;
      this.setChanged();
    }
  }
  get resolution() {
    return this.luminancePass.resolution.width;
  }
  set resolution(value) {
    const exponent = Math.max(0, Math.ceil(Math.log2(value)));
    const size = Math.pow(2, exponent);
    this.luminancePass.resolution.width = size;
    this.luminancePass.resolution.height = size;
    this.adaptiveLuminancePass.mipLevel1x1 = exponent;
  }
  get adaptive() {
    return this.defines.has("ADAPTIVE");
  }
  set adaptive(value) {
    this.mode = value ? ToneMappingMode.REINHARD2_ADAPTIVE : ToneMappingMode.REINHARD2;
  }
  get adaptationRate() {
    return this.adaptiveLuminancePass.adaptationRate;
  }
  set adaptationRate(value) {
    this.adaptiveLuminancePass.adaptationRate = value;
  }
  get distinction() {
    console.warn(this.name, "The distinction field has been removed.");
    return 1;
  }
  set distinction(value) {
    console.warn(this.name, "The distinction field has been removed.");
  }
  update(renderer, inputBuffer, deltaTime) {
    if (this.mode === ToneMappingMode.REINHARD2_ADAPTIVE) {
      this.luminancePass.render(renderer, inputBuffer);
      this.adaptiveLuminancePass.render(renderer, null, null, deltaTime);
    }
  }
  initialize(renderer, alpha, frameBufferType) {
    this.adaptiveLuminancePass.initialize(renderer, alpha, frameBufferType);
  }
};
var ToneMappingMode = {
  REINHARD: 0,
  REINHARD2: 1,
  REINHARD2_ADAPTIVE: 2,
  OPTIMIZED_CINEON: 3,
  ACES_FILMIC: 4
};

// src/effects/VignetteEffect.js
import {Uniform as Uniform43} from "../build/three.module.js";

// src/effects/glsl/vignette/shader.frag
var shader_default77 = "uniform float offset;uniform float darkness;void mainImage(const in vec4 inputColor,const in vec2 uv,out vec4 outputColor){const vec2 center=vec2(0.5);vec3 color=inputColor.rgb;\n#ifdef ESKIL\nvec2 coord=(uv-center)*vec2(offset);color=mix(color,vec3(1.0-darkness),dot(coord,coord));\n#else\nfloat d=distance(uv,center);color*=smoothstep(0.8,offset*0.799,d*(darkness+offset));\n#endif\noutputColor=vec4(color,inputColor.a);}";

// src/effects/VignetteEffect.js
var VignetteEffect = class extends Effect {
  constructor(options = {}) {
    const settings = Object.assign({
      blendFunction: BlendFunction.NORMAL,
      eskil: false,
      offset: 0.5,
      darkness: 0.5
    }, options);
    super("VignetteEffect", shader_default77, {
      blendFunction: settings.blendFunction,
      uniforms: new Map([
        ["offset", new Uniform43(settings.offset)],
        ["darkness", new Uniform43(settings.darkness)]
      ])
    });
    this.eskil = settings.eskil;
  }
  get eskil() {
    return this.defines.has("ESKIL");
  }
  set eskil(value) {
    if (this.eskil !== value) {
      if (value) {
        this.defines.set("ESKIL", "1");
      } else {
        this.defines.delete("ESKIL");
      }
      this.setChanged();
    }
  }
};

// src/images/lut/TetrahedralUpscaler.js
var P = [
  new Float32Array(3),
  new Float32Array(3)
];
var C = [
  new Float32Array(3),
  new Float32Array(3),
  new Float32Array(3),
  new Float32Array(3)
];
var T = [
  [
    new Float32Array([0, 0, 0]),
    new Float32Array([1, 0, 0]),
    new Float32Array([1, 1, 0]),
    new Float32Array([1, 1, 1])
  ],
  [
    new Float32Array([0, 0, 0]),
    new Float32Array([1, 0, 0]),
    new Float32Array([1, 0, 1]),
    new Float32Array([1, 1, 1])
  ],
  [
    new Float32Array([0, 0, 0]),
    new Float32Array([0, 0, 1]),
    new Float32Array([1, 0, 1]),
    new Float32Array([1, 1, 1])
  ],
  [
    new Float32Array([0, 0, 0]),
    new Float32Array([0, 1, 0]),
    new Float32Array([1, 1, 0]),
    new Float32Array([1, 1, 1])
  ],
  [
    new Float32Array([0, 0, 0]),
    new Float32Array([0, 1, 0]),
    new Float32Array([0, 1, 1]),
    new Float32Array([1, 1, 1])
  ],
  [
    new Float32Array([0, 0, 0]),
    new Float32Array([0, 0, 1]),
    new Float32Array([0, 1, 1]),
    new Float32Array([1, 1, 1])
  ]
];
function calculateTetrahedronVolume(a, b, c2, d) {
  const bcX = c2[0] - b[0];
  const bcY = c2[1] - b[1];
  const bcZ = c2[2] - b[2];
  const baX = a[0] - b[0];
  const baY = a[1] - b[1];
  const baZ = a[2] - b[2];
  const crossX = bcY * baZ - bcZ * baY;
  const crossY = bcZ * baX - bcX * baZ;
  const crossZ = bcX * baY - bcY * baX;
  const length = Math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
  const triangleArea = length * 0.5;
  const normalX = crossX / length;
  const normalY = crossY / length;
  const normalZ = crossZ / length;
  const constant = -(a[0] * normalX + a[1] * normalY + a[2] * normalZ);
  const dot = d[0] * normalX + d[1] * normalY + d[2] * normalZ;
  const height = Math.abs(dot + constant);
  return height * triangleArea / 3;
}
function sample(data, size, x, y, z, color2) {
  const i3 = (x + y * size + z * size * size) * 3;
  color2[0] = data[i3 + 0];
  color2[1] = data[i3 + 1];
  color2[2] = data[i3 + 2];
}
function tetrahedralSample(data, size, u, v3, w, color2) {
  const px = u * (size - 1);
  const py = v3 * (size - 1);
  const pz = w * (size - 1);
  const minX = Math.floor(px);
  const minY = Math.floor(py);
  const minZ = Math.floor(pz);
  const maxX = Math.ceil(px);
  const maxY = Math.ceil(py);
  const maxZ = Math.ceil(pz);
  const su = px - minX;
  const sv = py - minY;
  const sw = pz - minZ;
  if (minX === px && minY === py && minZ === pz) {
    sample(data, size, px, py, pz, color2);
  } else {
    let vertices;
    if (su >= sv && sv >= sw) {
      vertices = T[0];
    } else if (su >= sw && sw >= sv) {
      vertices = T[1];
    } else if (sw >= su && su >= sv) {
      vertices = T[2];
    } else if (sv >= su && su >= sw) {
      vertices = T[3];
    } else if (sv >= sw && sw >= su) {
      vertices = T[4];
    } else if (sw >= sv && sv >= su) {
      vertices = T[5];
    }
    const [P0, P1, P2, P3] = vertices;
    const coords = P[0];
    coords[0] = su;
    coords[1] = sv;
    coords[2] = sw;
    const tmp = P[1];
    const diffX = maxX - minX;
    const diffY = maxY - minY;
    const diffZ = maxZ - minZ;
    tmp[0] = diffX * P0[0] + minX;
    tmp[1] = diffY * P0[1] + minY;
    tmp[2] = diffZ * P0[2] + minZ;
    sample(data, size, tmp[0], tmp[1], tmp[2], C[0]);
    tmp[0] = diffX * P1[0] + minX;
    tmp[1] = diffY * P1[1] + minY;
    tmp[2] = diffZ * P1[2] + minZ;
    sample(data, size, tmp[0], tmp[1], tmp[2], C[1]);
    tmp[0] = diffX * P2[0] + minX;
    tmp[1] = diffY * P2[1] + minY;
    tmp[2] = diffZ * P2[2] + minZ;
    sample(data, size, tmp[0], tmp[1], tmp[2], C[2]);
    tmp[0] = diffX * P3[0] + minX;
    tmp[1] = diffY * P3[1] + minY;
    tmp[2] = diffZ * P3[2] + minZ;
    sample(data, size, tmp[0], tmp[1], tmp[2], C[3]);
    const V0 = calculateTetrahedronVolume(P1, P2, P3, coords) * 6;
    const V1 = calculateTetrahedronVolume(P0, P2, P3, coords) * 6;
    const V2 = calculateTetrahedronVolume(P0, P1, P3, coords) * 6;
    const V3 = calculateTetrahedronVolume(P0, P1, P2, coords) * 6;
    C[0][0] *= V0;
    C[0][1] *= V0;
    C[0][2] *= V0;
    C[1][0] *= V1;
    C[1][1] *= V1;
    C[1][2] *= V1;
    C[2][0] *= V2;
    C[2][1] *= V2;
    C[2][2] *= V2;
    C[3][0] *= V3;
    C[3][1] *= V3;
    C[3][2] *= V3;
    color2[0] = C[0][0] + C[1][0] + C[2][0] + C[3][0];
    color2[1] = C[0][1] + C[1][1] + C[2][1] + C[3][1];
    color2[2] = C[0][2] + C[1][2] + C[2][2] + C[3][2];
  }
}
var TetrahedralUpscaler = class {
  static expand(data, size) {
    const originalSize = Math.cbrt(data.length / 3);
    const rgb = new Float32Array(3);
    const array = new data.constructor(size ** 3 * 3);
    const s = 1 / (size - 1);
    for (let z = 0; z < size; ++z) {
      for (let y = 0; y < size; ++y) {
        for (let x = 0; x < size; ++x) {
          const u = x * s;
          const v3 = y * s;
          const w = z * s;
          const i3 = Math.round(x + y * size + z * size * size) * 3;
          tetrahedralSample(data, originalSize, u, v3, w, rgb);
          array[i3 + 0] = rgb[0];
          array[i3 + 1] = rgb[1];
          array[i3 + 2] = rgb[2];
        }
      }
    }
    return array;
  }
};

// src/images/smaa/SMAAAreaImageData.js
var area = [
  new Float32Array(2),
  new Float32Array(2)
];
var ORTHOGONAL_SIZE = 16;
var DIAGONAL_SIZE = 20;
var DIAGONAL_SAMPLES = 30;
var SMOOTH_MAX_DISTANCE = 32;
var orthogonalSubsamplingOffsets = new Float32Array([
  0,
  -0.25,
  0.25,
  -0.125,
  0.125,
  -0.375,
  0.375
]);
var diagonalSubsamplingOffsets = [
  new Float32Array([0, 0]),
  new Float32Array([0.25, -0.25]),
  new Float32Array([-0.25, 0.25]),
  new Float32Array([0.125, -0.125]),
  new Float32Array([-0.125, 0.125])
];
var orthogonalEdges = [
  new Uint8Array([0, 0]),
  new Uint8Array([3, 0]),
  new Uint8Array([0, 3]),
  new Uint8Array([3, 3]),
  new Uint8Array([1, 0]),
  new Uint8Array([4, 0]),
  new Uint8Array([1, 3]),
  new Uint8Array([4, 3]),
  new Uint8Array([0, 1]),
  new Uint8Array([3, 1]),
  new Uint8Array([0, 4]),
  new Uint8Array([3, 4]),
  new Uint8Array([1, 1]),
  new Uint8Array([4, 1]),
  new Uint8Array([1, 4]),
  new Uint8Array([4, 4])
];
var diagonalEdges = [
  new Uint8Array([0, 0]),
  new Uint8Array([1, 0]),
  new Uint8Array([0, 2]),
  new Uint8Array([1, 2]),
  new Uint8Array([2, 0]),
  new Uint8Array([3, 0]),
  new Uint8Array([2, 2]),
  new Uint8Array([3, 2]),
  new Uint8Array([0, 1]),
  new Uint8Array([1, 1]),
  new Uint8Array([0, 3]),
  new Uint8Array([1, 3]),
  new Uint8Array([2, 1]),
  new Uint8Array([3, 1]),
  new Uint8Array([2, 3]),
  new Uint8Array([3, 3])
];
function lerp(a, b, p) {
  return a + (b - a) * p;
}
function saturate(a) {
  return Math.min(Math.max(a, 0), 1);
}
function smoothArea(d) {
  const a1 = area[0];
  const a2 = area[1];
  const b1X = Math.sqrt(a1[0] * 2) * 0.5;
  const b1Y = Math.sqrt(a1[1] * 2) * 0.5;
  const b2X = Math.sqrt(a2[0] * 2) * 0.5;
  const b2Y = Math.sqrt(a2[1] * 2) * 0.5;
  const p = saturate(d / SMOOTH_MAX_DISTANCE);
  a1[0] = lerp(b1X, a1[0], p);
  a1[1] = lerp(b1Y, a1[1], p);
  a2[0] = lerp(b2X, a2[0], p);
  a2[1] = lerp(b2Y, a2[1], p);
}
function calculateOrthogonalArea(p1X, p1Y, p2X, p2Y, x, result) {
  const dX = p2X - p1X;
  const dY = p2Y - p1Y;
  const x1 = x;
  const x2 = x + 1;
  const y1 = p1Y + dY * (x1 - p1X) / dX;
  const y2 = p1Y + dY * (x2 - p1X) / dX;
  if (x1 >= p1X && x1 < p2X || x2 > p1X && x2 <= p2X) {
    if (Math.sign(y1) === Math.sign(y2) || Math.abs(y1) < 1e-4 || Math.abs(y2) < 1e-4) {
      const a = (y1 + y2) / 2;
      if (a < 0) {
        result[0] = Math.abs(a);
        result[1] = 0;
      } else {
        result[0] = 0;
        result[1] = Math.abs(a);
      }
    } else {
      const t = -p1Y * dX / dY + p1X;
      const tInt = Math.trunc(t);
      const a1 = t > p1X ? y1 * (t - tInt) / 2 : 0;
      const a2 = t < p2X ? y2 * (1 - (t - tInt)) / 2 : 0;
      const a = Math.abs(a1) > Math.abs(a2) ? a1 : -a2;
      if (a < 0) {
        result[0] = Math.abs(a1);
        result[1] = Math.abs(a2);
      } else {
        result[0] = Math.abs(a2);
        result[1] = Math.abs(a1);
      }
    }
  } else {
    result[0] = 0;
    result[1] = 0;
  }
  return result;
}
function calculateOrthogonalAreaForPattern(pattern, left, right, offset, result) {
  const a1 = area[0];
  const a2 = area[1];
  const o1 = 0.5 + offset;
  const o2 = 0.5 + offset - 1;
  const d = left + right + 1;
  switch (pattern) {
    case 0: {
      result[0] = 0;
      result[1] = 0;
      break;
    }
    case 1: {
      if (left <= right) {
        calculateOrthogonalArea(0, o2, d / 2, 0, left, result);
      } else {
        result[0] = 0;
        result[1] = 0;
      }
      break;
    }
    case 2: {
      if (left >= right) {
        calculateOrthogonalArea(d / 2, 0, d, o2, left, result);
      } else {
        result[0] = 0;
        result[1] = 0;
      }
      break;
    }
    case 3: {
      calculateOrthogonalArea(0, o2, d / 2, 0, left, a1);
      calculateOrthogonalArea(d / 2, 0, d, o2, left, a2);
      smoothArea(d, area);
      result[0] = a1[0] + a2[0];
      result[1] = a1[1] + a2[1];
      break;
    }
    case 4: {
      if (left <= right) {
        calculateOrthogonalArea(0, o1, d / 2, 0, left, result);
      } else {
        result[0] = 0;
        result[1] = 0;
      }
      break;
    }
    case 5: {
      result[0] = 0;
      result[1] = 0;
      break;
    }
    case 6: {
      if (Math.abs(offset) > 0) {
        calculateOrthogonalArea(0, o1, d, o2, left, a1);
        calculateOrthogonalArea(0, o1, d / 2, 0, left, a2);
        calculateOrthogonalArea(d / 2, 0, d, o2, left, result);
        a2[0] = a2[0] + result[0];
        a2[1] = a2[1] + result[1];
        result[0] = (a1[0] + a2[0]) / 2;
        result[1] = (a1[1] + a2[1]) / 2;
      } else {
        calculateOrthogonalArea(0, o1, d, o2, left, result);
      }
      break;
    }
    case 7: {
      calculateOrthogonalArea(0, o1, d, o2, left, result);
      break;
    }
    case 8: {
      if (left >= right) {
        calculateOrthogonalArea(d / 2, 0, d, o1, left, result);
      } else {
        result[0] = 0;
        result[1] = 0;
      }
      break;
    }
    case 9: {
      if (Math.abs(offset) > 0) {
        calculateOrthogonalArea(0, o2, d, o1, left, a1);
        calculateOrthogonalArea(0, o2, d / 2, 0, left, a2);
        calculateOrthogonalArea(d / 2, 0, d, o1, left, result);
        a2[0] = a2[0] + result[0];
        a2[1] = a2[1] + result[1];
        result[0] = (a1[0] + a2[0]) / 2;
        result[1] = (a1[1] + a2[1]) / 2;
      } else {
        calculateOrthogonalArea(0, o2, d, o1, left, result);
      }
      break;
    }
    case 10: {
      result[0] = 0;
      result[1] = 0;
      break;
    }
    case 11: {
      calculateOrthogonalArea(0, o2, d, o1, left, result);
      break;
    }
    case 12: {
      calculateOrthogonalArea(0, o1, d / 2, 0, left, a1);
      calculateOrthogonalArea(d / 2, 0, d, o1, left, a2);
      smoothArea(d, area);
      result[0] = a1[0] + a2[0];
      result[1] = a1[1] + a2[1];
      break;
    }
    case 13: {
      calculateOrthogonalArea(0, o2, d, o1, left, result);
      break;
    }
    case 14: {
      calculateOrthogonalArea(0, o1, d, o2, left, result);
      break;
    }
    case 15: {
      result[0] = 0;
      result[1] = 0;
      break;
    }
  }
  return result;
}
function isInsideArea(a1X, a1Y, a2X, a2Y, x, y) {
  let result = a1X === a2X && a1Y === a2Y;
  if (!result) {
    const xm = (a1X + a2X) / 2;
    const ym = (a1Y + a2Y) / 2;
    const a = a2Y - a1Y;
    const b = a1X - a2X;
    const c2 = a * (x - xm) + b * (y - ym);
    result = c2 > 0;
  }
  return result;
}
function calculateDiagonalAreaForPixel(a1X, a1Y, a2X, a2Y, pX, pY) {
  let n = 0;
  for (let y = 0; y < DIAGONAL_SAMPLES; ++y) {
    for (let x = 0; x < DIAGONAL_SAMPLES; ++x) {
      const offsetX = x / (DIAGONAL_SAMPLES - 1);
      const offsetY = y / (DIAGONAL_SAMPLES - 1);
      if (isInsideArea(a1X, a1Y, a2X, a2Y, pX + offsetX, pY + offsetY)) {
        ++n;
      }
    }
  }
  return n / (DIAGONAL_SAMPLES * DIAGONAL_SAMPLES);
}
function calculateDiagonalArea(pattern, a1X, a1Y, a2X, a2Y, left, offset, result) {
  const e = diagonalEdges[pattern];
  const e1 = e[0];
  const e2 = e[1];
  if (e1 > 0) {
    a1X += offset[0];
    a1Y += offset[1];
  }
  if (e2 > 0) {
    a2X += offset[0];
    a2Y += offset[1];
  }
  result[0] = 1 - calculateDiagonalAreaForPixel(a1X, a1Y, a2X, a2Y, 1 + left, 0 + left);
  result[1] = calculateDiagonalAreaForPixel(a1X, a1Y, a2X, a2Y, 1 + left, 1 + left);
  return result;
}
function calculateDiagonalAreaForPattern(pattern, left, right, offset, result) {
  const a1 = area[0];
  const a2 = area[1];
  const d = left + right + 1;
  switch (pattern) {
    case 0: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 1: {
      calculateDiagonalArea(pattern, 1, 0, 0 + d, 0 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 2: {
      calculateDiagonalArea(pattern, 0, 0, 1 + d, 0 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 3: {
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, result);
      break;
    }
    case 4: {
      calculateDiagonalArea(pattern, 1, 1, 0 + d, 0 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 5: {
      calculateDiagonalArea(pattern, 1, 1, 0 + d, 0 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 6: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 0 + d, left, offset, result);
      break;
    }
    case 7: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 0 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 8: {
      calculateDiagonalArea(pattern, 0, 0, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 1 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 9: {
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 1 + d, left, offset, result);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 1 + d, left, offset, result);
      break;
    }
    case 10: {
      calculateDiagonalArea(pattern, 0, 0, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 11: {
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 12: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 1 + d, left, offset, result);
      break;
    }
    case 13: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 1 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 14: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
    case 15: {
      calculateDiagonalArea(pattern, 1, 1, 1 + d, 1 + d, left, offset, a1);
      calculateDiagonalArea(pattern, 1, 0, 1 + d, 0 + d, left, offset, a2);
      result[0] = (a1[0] + a2[0]) / 2;
      result[1] = (a1[1] + a2[1]) / 2;
      break;
    }
  }
  return result;
}
function generatePatterns(patterns, offset, orthogonal) {
  const result = new Float32Array(2);
  for (let i = 0, l = patterns.length; i < l; ++i) {
    const pattern = patterns[i];
    const data = pattern.data;
    const size = pattern.width;
    for (let y = 0; y < size; ++y) {
      for (let x = 0; x < size; ++x) {
        if (orthogonal) {
          calculateOrthogonalAreaForPattern(i, x, y, offset, result);
        } else {
          calculateDiagonalAreaForPattern(i, x, y, offset, result);
        }
        const c2 = (y * size + x) * 2;
        data[c2] = result[0] * 255;
        data[c2 + 1] = result[1] * 255;
      }
    }
  }
}
function assemble(baseX, baseY, patterns, edges2, size, orthogonal, target) {
  const dstData = target.data;
  const dstWidth = target.width;
  for (let i = 0, l = patterns.length; i < l; ++i) {
    const edge = edges2[i];
    const pattern = patterns[i];
    const srcData = pattern.data;
    const srcWidth = pattern.width;
    for (let y = 0; y < size; ++y) {
      for (let x = 0; x < size; ++x) {
        const pX = edge[0] * size + baseX + x;
        const pY = edge[1] * size + baseY + y;
        const c2 = (pY * dstWidth + pX) * 4;
        const d = orthogonal ? (y * y * srcWidth + x * x) * 2 : (y * srcWidth + x) * 2;
        dstData[c2] = srcData[d];
        dstData[c2 + 1] = srcData[d + 1];
        dstData[c2 + 2] = 0;
        dstData[c2 + 3] = 255;
      }
    }
  }
}
var SMAAAreaImageData = class {
  static generate() {
    const width = 2 * 5 * ORTHOGONAL_SIZE;
    const height = orthogonalSubsamplingOffsets.length * 5 * ORTHOGONAL_SIZE;
    const data = new Uint8ClampedArray(width * height * 4);
    const result = new RawImageData(width, height, data);
    const orthogonalPatternSize = Math.pow(ORTHOGONAL_SIZE - 1, 2) + 1;
    const diagonalPatternSize = DIAGONAL_SIZE;
    const orthogonalPatterns = [];
    const diagonalPatterns = [];
    for (let i = 3, l = data.length; i < l; i += 4) {
      data[i] = 255;
    }
    for (let i = 0; i < 16; ++i) {
      orthogonalPatterns.push(new RawImageData(orthogonalPatternSize, orthogonalPatternSize, new Uint8ClampedArray(orthogonalPatternSize * orthogonalPatternSize * 2), 2));
      diagonalPatterns.push(new RawImageData(diagonalPatternSize, diagonalPatternSize, new Uint8ClampedArray(diagonalPatternSize * diagonalPatternSize * 2), 2));
    }
    for (let i = 0, l = orthogonalSubsamplingOffsets.length; i < l; ++i) {
      generatePatterns(orthogonalPatterns, orthogonalSubsamplingOffsets[i], true);
      assemble(0, 5 * ORTHOGONAL_SIZE * i, orthogonalPatterns, orthogonalEdges, ORTHOGONAL_SIZE, true, result);
    }
    for (let i = 0, l = diagonalSubsamplingOffsets.length; i < l; ++i) {
      generatePatterns(diagonalPatterns, diagonalSubsamplingOffsets[i], false);
      assemble(5 * ORTHOGONAL_SIZE, 4 * DIAGONAL_SIZE * i, diagonalPatterns, diagonalEdges, DIAGONAL_SIZE, false, result);
    }
    return result;
  }
};

// src/images/smaa/SMAAImageGenerator.js
import {LoadingManager} from "../build/three.module.js";

// tmp/smaa/worker.txt
var worker_default2 = '(()=>{function q(t,a,s){let i=document.createElementNS("http://www.w3.org/1999/xhtml","canvas"),n=i.getContext("2d");if(i.width=t,i.height=a,s instanceof Image)n.drawImage(s,0,0);else{let o=n.createImageData(t,a);o.data.set(s),n.putImageData(o,0,0)}return i}var k=class{constructor(a=0,s=0,i=null){this.width=a,this.height=s,this.data=i}toCanvas(){return typeof document=="undefined"?null:q(this.width,this.height,this.data)}static from(a){let{width:s,height:i}=a,n;if(a instanceof Image){let o=q(s,i,a);o!==null&&(n=o.getContext("2d").getImageData(0,0,s,i).data)}else n=a.data;return new k(s,i,n)}};var F=[new Float32Array(2),new Float32Array(2)],I=16,P=20,x=30,z=32,E=new Float32Array([0,-.25,.25,-.125,.125,-.375,.375]),W=[new Float32Array([0,0]),new Float32Array([.25,-.25]),new Float32Array([-.25,.25]),new Float32Array([.125,-.125]),new Float32Array([-.125,.125])],B=[new Uint8Array([0,0]),new Uint8Array([3,0]),new Uint8Array([0,3]),new Uint8Array([3,3]),new Uint8Array([1,0]),new Uint8Array([4,0]),new Uint8Array([1,3]),new Uint8Array([4,3]),new Uint8Array([0,1]),new Uint8Array([3,1]),new Uint8Array([0,4]),new Uint8Array([3,4]),new Uint8Array([1,1]),new Uint8Array([4,1]),new Uint8Array([1,4]),new Uint8Array([4,4])],_=[new Uint8Array([0,0]),new Uint8Array([1,0]),new Uint8Array([0,2]),new Uint8Array([1,2]),new Uint8Array([2,0]),new Uint8Array([3,0]),new Uint8Array([2,2]),new Uint8Array([3,2]),new Uint8Array([0,1]),new Uint8Array([1,1]),new Uint8Array([0,3]),new Uint8Array([1,3]),new Uint8Array([2,1]),new Uint8Array([3,1]),new Uint8Array([2,3]),new Uint8Array([3,3])];function O(t,a,s){return t+(a-t)*s}function J(t){return Math.min(Math.max(t,0),1)}function G(t){let a=F[0],s=F[1],i=Math.sqrt(a[0]*2)*.5,n=Math.sqrt(a[1]*2)*.5,o=Math.sqrt(s[0]*2)*.5,r=Math.sqrt(s[1]*2)*.5,c=J(t/z);a[0]=O(i,a[0],c),a[1]=O(n,a[1],c),s[0]=O(o,s[0],c),s[1]=O(r,s[1],c)}function y(t,a,s,i,n,o){let r=s-t,c=i-a,w=n,e=n+1,A=a+c*(w-t)/r,g=a+c*(e-t)/r;if(w>=t&&w<s||e>t&&e<=s)if(Math.sign(A)===Math.sign(g)||Math.abs(A)<1e-4||Math.abs(g)<1e-4){let b=(A+g)/2;b<0?(o[0]=Math.abs(b),o[1]=0):(o[0]=0,o[1]=Math.abs(b))}else{let b=-a*r/c+t,M=Math.trunc(b),U=b>t?A*(b-M)/2:0,m=b<s?g*(1-(b-M))/2:0;(Math.abs(U)>Math.abs(m)?U:-m)<0?(o[0]=Math.abs(U),o[1]=Math.abs(m)):(o[0]=Math.abs(m),o[1]=Math.abs(U))}else o[0]=0,o[1]=0;return o}function K(t,a,s,i,n){let o=F[0],r=F[1],c=.5+i,w=.5+i-1,e=a+s+1;switch(t){case 0:{n[0]=0,n[1]=0;break}case 1:{a<=s?y(0,w,e/2,0,a,n):(n[0]=0,n[1]=0);break}case 2:{a>=s?y(e/2,0,e,w,a,n):(n[0]=0,n[1]=0);break}case 3:{y(0,w,e/2,0,a,o),y(e/2,0,e,w,a,r),G(e,F),n[0]=o[0]+r[0],n[1]=o[1]+r[1];break}case 4:{a<=s?y(0,c,e/2,0,a,n):(n[0]=0,n[1]=0);break}case 5:{n[0]=0,n[1]=0;break}case 6:{Math.abs(i)>0?(y(0,c,e,w,a,o),y(0,c,e/2,0,a,r),y(e/2,0,e,w,a,n),r[0]=r[0]+n[0],r[1]=r[1]+n[1],n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2):y(0,c,e,w,a,n);break}case 7:{y(0,c,e,w,a,n);break}case 8:{a>=s?y(e/2,0,e,c,a,n):(n[0]=0,n[1]=0);break}case 9:{Math.abs(i)>0?(y(0,w,e,c,a,o),y(0,w,e/2,0,a,r),y(e/2,0,e,c,a,n),r[0]=r[0]+n[0],r[1]=r[1]+n[1],n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2):y(0,w,e,c,a,n);break}case 10:{n[0]=0,n[1]=0;break}case 11:{y(0,w,e,c,a,n);break}case 12:{y(0,c,e/2,0,a,o),y(e/2,0,e,c,a,r),G(e,F),n[0]=o[0]+r[0],n[1]=o[1]+r[1];break}case 13:{y(0,w,e,c,a,n);break}case 14:{y(0,c,e,w,a,n);break}case 15:{n[0]=0,n[1]=0;break}}return n}function Q(t,a,s,i,n,o){let r=t===s&&a===i;if(!r){let c=(t+s)/2,w=(a+i)/2,e=i-a,A=t-s;r=e*(n-c)+A*(o-w)>0}return r}function H(t,a,s,i,n,o){let r=0;for(let c=0;c<x;++c)for(let w=0;w<x;++w){let e=w/(x-1),A=c/(x-1);Q(t,a,s,i,n+e,o+A)&&++r}return r/(x*x)}function h(t,a,s,i,n,o,r,c){let w=_[t],e=w[0],A=w[1];return e>0&&(a+=r[0],s+=r[1]),A>0&&(i+=r[0],n+=r[1]),c[0]=1-H(a,s,i,n,1+o,0+o),c[1]=H(a,s,i,n,1+o,1+o),c}function V(t,a,s,i,n){let o=F[0],r=F[1],c=a+s+1;switch(t){case 0:{h(t,1,1,1+c,1+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 1:{h(t,1,0,0+c,0+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 2:{h(t,0,0,1+c,0+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 3:{h(t,1,0,1+c,0+c,a,i,n);break}case 4:{h(t,1,1,0+c,0+c,a,i,o),h(t,1,1,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 5:{h(t,1,1,0+c,0+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 6:{h(t,1,1,1+c,0+c,a,i,n);break}case 7:{h(t,1,1,1+c,0+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 8:{h(t,0,0,1+c,1+c,a,i,o),h(t,1,0,1+c,1+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 9:{h(t,1,0,1+c,1+c,a,i,n),h(t,1,0,1+c,1+c,a,i,n);break}case 10:{h(t,0,0,1+c,1+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 11:{h(t,1,0,1+c,1+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 12:{h(t,1,1,1+c,1+c,a,i,n);break}case 13:{h(t,1,1,1+c,1+c,a,i,o),h(t,1,0,1+c,1+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 14:{h(t,1,1,1+c,1+c,a,i,o),h(t,1,1,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}case 15:{h(t,1,1,1+c,1+c,a,i,o),h(t,1,0,1+c,0+c,a,i,r),n[0]=(o[0]+r[0])/2,n[1]=(o[1]+r[1])/2;break}}return n}function T(t,a,s){let i=new Float32Array(2);for(let n=0,o=t.length;n<o;++n){let r=t[n],c=r.data,w=r.width;for(let e=0;e<w;++e)for(let A=0;A<w;++A){s?K(n,A,e,a,i):V(n,A,e,a,i);let g=(e*w+A)*2;c[g]=i[0]*255,c[g+1]=i[1]*255}}}function Z(t,a,s,i,n,o,r){let c=r.data,w=r.width;for(let e=0,A=s.length;e<A;++e){let g=i[e],b=s[e],M=b.data,U=b.width;for(let m=0;m<n;++m)for(let D=0;D<n;++D){let j=g[0]*n+t+D,S=((g[1]*n+a+m)*w+j)*4,R=o?(m*m*U+D*D)*2:(m*U+D)*2;c[S]=M[R],c[S+1]=M[R+1],c[S+2]=0,c[S+3]=255}}}var v=class{static generate(){let a=2*5*I,s=E.length*5*I,i=new Uint8ClampedArray(a*s*4),n=new k(a,s,i),o=Math.pow(I-1,2)+1,r=P,c=[],w=[];for(let e=3,A=i.length;e<A;e+=4)i[e]=255;for(let e=0;e<16;++e)c.push(new k(o,o,new Uint8ClampedArray(o*o*2),2)),w.push(new k(r,r,new Uint8ClampedArray(r*r*2),2));for(let e=0,A=E.length;e<A;++e)T(c,E[e],!0),Z(0,5*I*e,c,B,I,!0,n);for(let e=0,A=W.length;e<A;++e)T(w,W[e],!1),Z(5*I,4*P*e,w,_,P,!1,n);return n}};var C=new Map([[d(0,0,0,0),new Float32Array([0,0,0,0])],[d(0,0,0,1),new Float32Array([0,0,0,1])],[d(0,0,1,0),new Float32Array([0,0,1,0])],[d(0,0,1,1),new Float32Array([0,0,1,1])],[d(0,1,0,0),new Float32Array([0,1,0,0])],[d(0,1,0,1),new Float32Array([0,1,0,1])],[d(0,1,1,0),new Float32Array([0,1,1,0])],[d(0,1,1,1),new Float32Array([0,1,1,1])],[d(1,0,0,0),new Float32Array([1,0,0,0])],[d(1,0,0,1),new Float32Array([1,0,0,1])],[d(1,0,1,0),new Float32Array([1,0,1,0])],[d(1,0,1,1),new Float32Array([1,0,1,1])],[d(1,1,0,0),new Float32Array([1,1,0,0])],[d(1,1,0,1),new Float32Array([1,1,0,1])],[d(1,1,1,0),new Float32Array([1,1,1,0])],[d(1,1,1,1),new Float32Array([1,1,1,1])]]);function L(t,a,s){return t+(a-t)*s}function d(t,a,s,i){let n=L(t,a,1-.25),o=L(s,i,1-.25);return L(n,o,1-.125)}function $(t,a){let s=0;return a[3]===1&&(s+=1),s===1&&a[2]===1&&t[1]!==1&&t[3]!==1&&(s+=1),s}function Y(t,a){let s=0;return a[3]===1&&t[1]!==1&&t[3]!==1&&(s+=1),s===1&&a[2]===1&&t[0]!==1&&t[2]!==1&&(s+=1),s}var N=class{static generate(){let a=66,s=33,i=a/2,n=64,o=16,r=new Uint8ClampedArray(a*s),c=new Uint8ClampedArray(n*o*4);for(let w=0;w<s;++w)for(let e=0;e<a;++e){let A=.03125*e,g=.03125*w;if(C.has(A)&&C.has(g)){let b=C.get(A),M=C.get(g),U=w*a+e;r[U]=127*$(b,M),r[U+i]=127*Y(b,M)}}for(let w=0,e=s-o;e<s;++e)for(let A=0;A<n;++A,w+=4)c[w]=r[e*a+A],c[w+3]=255;return new k(n,o,c)}};self.addEventListener("message",t=>{let a=v.generate(),s=N.generate();postMessage({areaImageData:a,searchImageData:s},[a.data.buffer,s.data.buffer]),close()});})();\n';

// src/images/smaa/SMAAImageGenerator.js
function generate(useCache = true) {
  const workerURL = URL.createObjectURL(new Blob([worker_default2], {
    type: "text/javascript"
  }));
  const worker = new Worker(workerURL);
  URL.revokeObjectURL(workerURL);
  return new Promise((resolve, reject) => {
    worker.addEventListener("error", (event) => reject(event.error));
    worker.addEventListener("message", (event) => {
      const searchImageData = RawImageData.from(event.data.searchImageData);
      const areaImageData = RawImageData.from(event.data.areaImageData);
      const urls = [
        searchImageData.toCanvas().toDataURL("image/png", 1),
        areaImageData.toCanvas().toDataURL("image/png", 1)
      ];
      if (useCache) {
        localStorage.setItem("smaa-search", urls[0]);
        localStorage.setItem("smaa-area", urls[1]);
      }
      resolve(urls);
    });
    worker.postMessage(null);
  });
}
var SMAAImageGenerator = class {
  constructor() {
    this.disableCache = false;
  }
  generate() {
    const useCache = !this.disableCache && window.localStorage !== void 0;
    const cachedURLs = useCache ? [
      localStorage.getItem("smaa-search"),
      localStorage.getItem("smaa-area")
    ] : [null, null];
    const promise = cachedURLs[0] !== null && cachedURLs[1] !== null ? Promise.resolve(cachedURLs) : generate(useCache);
    return promise.then((urls) => {
      return new Promise((resolve, reject) => {
        const searchImage = new Image();
        const areaImage = new Image();
        const manager = new LoadingManager();
        manager.onLoad = () => resolve([searchImage, areaImage]);
        manager.onError = reject;
        searchImage.addEventListener("error", (e) => manager.itemError("smaa-search"));
        areaImage.addEventListener("error", (e) => manager.itemError("smaa-area"));
        searchImage.addEventListener("load", () => manager.itemEnd("smaa-search"));
        areaImage.addEventListener("load", () => manager.itemEnd("smaa-area"));
        manager.itemStart("smaa-search");
        manager.itemStart("smaa-area");
        searchImage.src = urls[0];
        areaImage.src = urls[1];
      });
    });
  }
};

// src/images/smaa/SMAASearchImageData.js
var edges = new Map([
  [bilinear(0, 0, 0, 0), new Float32Array([0, 0, 0, 0])],
  [bilinear(0, 0, 0, 1), new Float32Array([0, 0, 0, 1])],
  [bilinear(0, 0, 1, 0), new Float32Array([0, 0, 1, 0])],
  [bilinear(0, 0, 1, 1), new Float32Array([0, 0, 1, 1])],
  [bilinear(0, 1, 0, 0), new Float32Array([0, 1, 0, 0])],
  [bilinear(0, 1, 0, 1), new Float32Array([0, 1, 0, 1])],
  [bilinear(0, 1, 1, 0), new Float32Array([0, 1, 1, 0])],
  [bilinear(0, 1, 1, 1), new Float32Array([0, 1, 1, 1])],
  [bilinear(1, 0, 0, 0), new Float32Array([1, 0, 0, 0])],
  [bilinear(1, 0, 0, 1), new Float32Array([1, 0, 0, 1])],
  [bilinear(1, 0, 1, 0), new Float32Array([1, 0, 1, 0])],
  [bilinear(1, 0, 1, 1), new Float32Array([1, 0, 1, 1])],
  [bilinear(1, 1, 0, 0), new Float32Array([1, 1, 0, 0])],
  [bilinear(1, 1, 0, 1), new Float32Array([1, 1, 0, 1])],
  [bilinear(1, 1, 1, 0), new Float32Array([1, 1, 1, 0])],
  [bilinear(1, 1, 1, 1), new Float32Array([1, 1, 1, 1])]
]);
function lerp2(a, b, p) {
  return a + (b - a) * p;
}
function bilinear(e0, e1, e2, e3) {
  const a = lerp2(e0, e1, 1 - 0.25);
  const b = lerp2(e2, e3, 1 - 0.25);
  return lerp2(a, b, 1 - 0.125);
}
function deltaLeft(left, top) {
  let d = 0;
  if (top[3] === 1) {
    d += 1;
  }
  if (d === 1 && top[2] === 1 && left[1] !== 1 && left[3] !== 1) {
    d += 1;
  }
  return d;
}
function deltaRight(left, top) {
  let d = 0;
  if (top[3] === 1 && left[1] !== 1 && left[3] !== 1) {
    d += 1;
  }
  if (d === 1 && top[2] === 1 && left[0] !== 1 && left[2] !== 1) {
    d += 1;
  }
  return d;
}
var SMAASearchImageData = class {
  static generate() {
    const width = 66;
    const height = 33;
    const halfWidth = width / 2;
    const croppedWidth = 64;
    const croppedHeight = 16;
    const data = new Uint8ClampedArray(width * height);
    const croppedData = new Uint8ClampedArray(croppedWidth * croppedHeight * 4);
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const s = 0.03125 * x;
        const t = 0.03125 * y;
        if (edges.has(s) && edges.has(t)) {
          const e1 = edges.get(s);
          const e2 = edges.get(t);
          const i = y * width + x;
          data[i] = 127 * deltaLeft(e1, e2);
          data[i + halfWidth] = 127 * deltaRight(e1, e2);
        }
      }
    }
    for (let i = 0, y = height - croppedHeight; y < height; ++y) {
      for (let x = 0; x < croppedWidth; ++x, i += 4) {
        croppedData[i] = data[y * width + x];
        croppedData[i + 3] = 255;
      }
    }
    return new RawImageData(croppedWidth, croppedHeight, croppedData);
  }
};

// src/loaders/LUT3dlLoader.js
import {FileLoader, Loader, LoadingManager as LoadingManager2, sRGBEncoding as sRGBEncoding4} from "../build/three.module.js";
var LUT3dlLoader = class extends Loader {
  load(url, onLoad = () => {
  }, onProgress = () => {
  }, onError = null) {
    const externalManager = this.manager;
    const internalManager = new LoadingManager2();
    const loader = new FileLoader(internalManager);
    loader.setPath(this.path);
    loader.setResponseType("text");
    return new Promise((resolve, reject) => {
      internalManager.onError = (url2) => {
        externalManager.itemError(url2);
        if (onError !== null) {
          onError(`Failed to load ${url2}`);
          resolve();
        } else {
          reject(`Failed to load ${url2}`);
        }
      };
      externalManager.itemStart(url);
      loader.load(url, (data) => {
        try {
          const result = this.parse(data);
          externalManager.itemEnd(url);
          onLoad(result);
          resolve(result);
        } catch (e) {
          console.error(e);
          internalManager.onError(url);
        }
      }, onProgress);
    });
  }
  parse(input) {
    const regExpGridInfo = /^[\d ]+$/m;
    const regExpDataPoints = /^([\d.]+) +([\d.]+) +([\d.]+) *$/gm;
    let result = regExpGridInfo.exec(input);
    if (result === null) {
      throw new Error("Missing grid information");
    }
    const gridLines = result[0].trim().split(/\s+/g).map((n) => Number(n));
    const gridStep = gridLines[1] - gridLines[0];
    const size = gridLines.length;
    for (let i = 1, l = gridLines.length; i < l; ++i) {
      if (gridStep !== gridLines[i] - gridLines[i - 1]) {
        throw new Error("Inconsistent grid size");
      }
    }
    const data = new Float32Array(size ** 3 * 3);
    let maxValue = 0;
    let index = 0;
    while ((result = regExpDataPoints.exec(input)) !== null) {
      const r = Number(result[1]);
      const g = Number(result[2]);
      const b = Number(result[3]);
      maxValue = Math.max(maxValue, r, g, b);
      const bLayer = index % size;
      const gLayer = Math.floor(index / size) % size;
      const rLayer = Math.floor(index / (size * size)) % size;
      const d3 = (bLayer * size * size + gLayer * size + rLayer) * 3;
      data[d3 + 0] = r;
      data[d3 + 1] = g;
      data[d3 + 2] = b;
      ++index;
    }
    const bits = Math.ceil(Math.log2(maxValue));
    const maxBitValue = Math.pow(2, bits);
    for (let i = 0, l = data.length; i < l; ++i) {
      data[i] /= maxBitValue;
    }
    const lut = new LookupTexture3D(data, size, size, size);
    lut.encoding = sRGBEncoding4;
    return lut;
  }
};

// src/loaders/LUTCubeLoader.js
import {FileLoader as FileLoader2, Loader as Loader2, LoadingManager as LoadingManager3, sRGBEncoding as sRGBEncoding5, Vector3 as Vector36} from "../build/three.module.js";
var LUTCubeLoader = class extends Loader2 {
  load(url, onLoad = () => {
  }, onProgress = () => {
  }, onError = null) {
    const externalManager = this.manager;
    const internalManager = new LoadingManager3();
    const loader = new FileLoader2(internalManager);
    loader.setPath(this.path);
    loader.setResponseType("text");
    return new Promise((resolve, reject) => {
      internalManager.onError = (url2) => {
        externalManager.itemError(url2);
        if (onError !== null) {
          onError(`Failed to load ${url2}`);
          resolve();
        } else {
          reject(`Failed to load ${url2}`);
        }
      };
      externalManager.itemStart(url);
      loader.load(url, (data) => {
        try {
          const result = this.parse(data);
          externalManager.itemEnd(url);
          onLoad(result);
          resolve(result);
        } catch (e) {
          console.error(e);
          internalManager.onError(url);
        }
      }, onProgress);
    });
  }
  parse(input) {
    const regExpTitle = /TITLE +"([^"]*)"/;
    const regExpSize = /LUT_3D_SIZE +(\d+)/;
    const regExpDomainMin = /DOMAIN_MIN +([\d.]+) +([\d.]+) +([\d.]+)/;
    const regExpDomainMax = /DOMAIN_MAX +([\d.]+) +([\d.]+) +([\d.]+)/;
    const regExpDataPoints = /^([\d.]+) +([\d.]+) +([\d.]+) *$/gm;
    let result = regExpTitle.exec(input);
    const title = result !== null ? result[1] : null;
    result = regExpSize.exec(input);
    if (result === null) {
      throw new Error("Missing LUT_3D_SIZE information");
    }
    const size = Number(result[1]);
    const data = new Float32Array(size ** 3 * 3);
    const domainMin = new Vector36(0, 0, 0);
    const domainMax = new Vector36(1, 1, 1);
    result = regExpDomainMin.exec(input);
    if (result !== null) {
      domainMin.set(Number(result[1]), Number(result[2]), Number(result[3]));
    }
    result = regExpDomainMax.exec(input);
    if (result !== null) {
      domainMax.set(Number(result[1]), Number(result[2]), Number(result[3]));
    }
    if (domainMin.x > domainMax.x || domainMin.y > domainMax.y || domainMin.z > domainMax.z) {
      domainMin.set(0, 0, 0);
      domainMax.set(1, 1, 1);
      throw new Error("Invalid input domain");
    }
    let i = 0;
    while ((result = regExpDataPoints.exec(input)) !== null) {
      data[i++] = Number(result[1]);
      data[i++] = Number(result[2]);
      data[i++] = Number(result[3]);
    }
    const lut = new LookupTexture3D(data, size, size, size);
    lut.encoding = sRGBEncoding5;
    lut.domainMin.copy(domainMin);
    lut.domainMax.copy(domainMax);
    if (title !== null) {
      lut.name = title;
    }
    return lut;
  }
};

// src/loaders/SMAAImageLoader.js
import {Loader as Loader3, LoadingManager as LoadingManager4} from "../build/three.module.js";
var SMAAImageLoader = class extends Loader3 {
  load(onLoad = () => {
  }, onError = null) {
    if (arguments.length === 4) {
      onLoad = arguments[1];
      onError = arguments[3];
    } else if (arguments.length === 3 || typeof arguments[0] !== "function") {
      onLoad = arguments[1];
      onError = null;
    }
    const externalManager = this.manager;
    const internalManager = new LoadingManager4();
    return new Promise((resolve, reject) => {
      const searchImage = new Image();
      const areaImage = new Image();
      internalManager.onError = (url) => {
        externalManager.itemError(url);
        if (onError !== null) {
          onError(`Failed to load ${url}`);
          resolve();
        } else {
          reject(`Failed to load ${url}`);
        }
      };
      internalManager.onLoad = () => {
        const result = [searchImage, areaImage];
        onLoad(result);
        resolve(result);
      };
      searchImage.addEventListener("error", (e) => {
        internalManager.itemError("smaa-search");
      });
      areaImage.addEventListener("error", (e) => {
        internalManager.itemError("smaa-area");
      });
      searchImage.addEventListener("load", () => {
        externalManager.itemEnd("smaa-search");
        internalManager.itemEnd("smaa-search");
      });
      areaImage.addEventListener("load", () => {
        externalManager.itemEnd("smaa-area");
        internalManager.itemEnd("smaa-area");
      });
      externalManager.itemStart("smaa-search");
      externalManager.itemStart("smaa-area");
      internalManager.itemStart("smaa-search");
      internalManager.itemStart("smaa-area");
      searchImage.src = searchImageDataURL_default;
      areaImage.src = areaImageDataURL_default;
    });
  }
};
export {
  AdaptiveLuminanceMaterial,
  AdaptiveLuminancePass,
  BlendFunction,
  BlendMode,
  BloomEffect,
  BlurPass,
  BokehEffect,
  BokehMaterial,
  BrightnessContrastEffect,
  ChromaticAberrationEffect,
  CircleOfConfusionMaterial,
  ClearMaskPass,
  ClearPass,
  ColorAverageEffect,
  ColorChannel,
  ColorDepthEffect,
  ColorEdgesMaterial,
  ConvolutionMaterial,
  CopyMaterial,
  DepthComparisonMaterial,
  DepthCopyMaterial,
  DepthCopyMode,
  DepthDownsamplingMaterial,
  DepthDownsamplingPass,
  DepthEffect,
  DepthMaskMaterial,
  DepthOfFieldEffect,
  DepthPass,
  DepthPickingPass,
  DepthSavePass,
  Disposable,
  DotScreenEffect,
  EdgeDetectionMaterial,
  EdgeDetectionMode,
  Effect,
  EffectAttribute,
  EffectComposer,
  EffectMaterial,
  EffectPass,
  GammaCorrectionEffect,
  GlitchEffect,
  GlitchMode,
  GodRaysEffect,
  GodRaysMaterial,
  GridEffect,
  HueSaturationEffect,
  Initializable,
  KernelSize,
  LUT3dlLoader,
  LUTCubeLoader,
  LUTEffect,
  LUTOperation,
  LambdaPass,
  LookupTexture3D,
  LuminanceMaterial,
  LuminancePass,
  MaskFunction,
  MaskMaterial,
  MaskPass,
  NoiseEffect,
  NoiseTexture,
  NormalPass,
  OutlineEdgesMaterial,
  OutlineEffect,
  OutlineMaterial,
  OverrideMaterialManager,
  Pass,
  PixelationEffect,
  PredicationMode,
  RawImageData,
  RealisticBokehEffect,
  RenderPass,
  Resizable,
  Resizer,
  SMAAAreaImageData,
  SMAAEffect,
  SMAAImageGenerator,
  SMAAImageLoader,
  SMAAPreset,
  SMAASearchImageData,
  SMAAWeightsMaterial,
  SSAOEffect,
  SSAOMaterial,
  SavePass,
  ScanlineEffect,
  Section,
  Selection,
  SelectiveBloomEffect,
  SepiaEffect,
  ShaderPass,
  ShockWaveEffect,
  TetrahedralUpscaler,
  TextureEffect,
  ToneMappingEffect,
  ToneMappingMode,
  VignetteEffect,
  WebGLExtension
};
