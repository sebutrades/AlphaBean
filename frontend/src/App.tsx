import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import AlphaBeanLogo from "./AlphaBeanLogo";
const API = "http://localhost:8000";

const DARK={bg:"#0a0e17",bgCard:"#111827",bgHover:"#1a2332",bgModal:"#0d1117",border:"#1e293b",borderLight:"#334155",text:"#f1f5f9",textDim:"#94a3b8",textMuted:"#64748b",accent:"#3b82f6",long:"#22c55e",longBg:"rgba(34,197,94,0.12)",short:"#ef4444",shortBg:"rgba(239,68,68,0.12)",gold:"#eab308",goldBg:"rgba(234,179,8,0.12)",purple:"#a855f7",chartBg:"#0a0e17",chartGrid:"#1e293b",chartText:"#64748b"};
const LIGHT={bg:"#f8fafc",bgCard:"#ffffff",bgHover:"#f1f5f9",bgModal:"#f1f5f9",border:"#e2e8f0",borderLight:"#cbd5e1",text:"#0f172a",textDim:"#64748b",textMuted:"#94a3b8",accent:"#3b82f6",long:"#16a34a",longBg:"rgba(22,163,74,0.08)",short:"#dc2626",shortBg:"rgba(220,38,38,0.08)",gold:"#ca8a04",goldBg:"rgba(202,138,4,0.08)",purple:"#7c3aed",chartBg:"#ffffff",chartGrid:"#f1f5f9",chartText:"#64748b"};
type Th=typeof DARK;

// ══════════════════════════════════════════════════════
// MINING LOADING ANIMATION
// ══════════════════════════════════════════════════════
function MiningAnimation({t,msg="Loading..."}:{t:Th;msg?:string}){
  return(
    <div style={{textAlign:"center",padding:"40px 20px"}}>
      <style>{`
        @keyframes pickSwing{0%,100%{transform:rotate(-35deg)}50%{transform:rotate(35deg)}}
        @keyframes sparkA{0%{opacity:0;transform:translate(0,0) scale(0)}30%{opacity:1;transform:translate(-20px,-30px) scale(1)}100%{opacity:0;transform:translate(-40px,-60px) scale(0)}}
        @keyframes sparkB{0%{opacity:0;transform:translate(0,0) scale(0)}30%{opacity:1;transform:translate(15px,-35px) scale(1)}100%{opacity:0;transform:translate(30px,-70px) scale(0)}}
        @keyframes sparkC{0%{opacity:0;transform:translate(0,0) scale(0)}30%{opacity:1;transform:translate(25px,-20px) scale(1)}100%{opacity:0;transform:translate(50px,-40px) scale(0)}}
        @keyframes rockShake{0%,100%{transform:translateX(0)}25%{transform:translateX(-2px)}75%{transform:translateX(2px)}}
        @keyframes dustUp{0%{opacity:0;transform:translateY(0) scale(0.5)}50%{opacity:0.4;transform:translateY(-15px) scale(1)}100%{opacity:0;transform:translateY(-30px) scale(1.5)}}
        @keyframes gemPulse{0%,100%{opacity:0.3;filter:brightness(1)}50%{opacity:1;filter:brightness(1.5)}}
        @keyframes progFill{0%{width:0%}100%{width:85%}}
        @keyframes dotJump{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
        @keyframes txtPulse{0%,100%{opacity:0.7}50%{opacity:1}}
        @keyframes crackDraw{0%{stroke-dashoffset:50;opacity:0}50%{stroke-dashoffset:0;opacity:0.6}100%{stroke-dashoffset:0;opacity:0}}
      `}</style>
      <svg width="400" height="260" viewBox="0 0 400 260" fill="none" style={{maxWidth:"100%"}}>
        {/* Surface */}
        <rect x="0" y="0" width="400" height="95" fill={t.bg}/>
        <path d="M0 95 Q40 90 80 95 Q120 100 160 95 Q200 90 240 95 Q280 100 320 95 Q360 90 400 95" fill="#4a3728" stroke="#5d4037" strokeWidth="2"/>
        {/* Dirt */}
        <rect x="0" y="95" width="400" height="65" fill="#3e2723"/>
        {/* Rock */}
        <rect x="0" y="160" width="400" height="100" fill="#2c1e14"/>
        {/* Rock textures */}
        <circle cx="60" cy="190" r="18" fill="#3e2723" opacity="0.5"/>
        <circle cx="340" cy="180" r="22" fill="#3e2723" opacity="0.4"/>
        <circle cx="200" cy="220" r="14" fill="#3e2723" opacity="0.5"/>
        <ellipse cx="120" cy="230" rx="28" ry="12" fill="#3e2723" opacity="0.3"/>
        <ellipse cx="300" cy="240" rx="20" ry="9" fill="#3e2723" opacity="0.3"/>
        {/* Gems */}
        <g style={{animation:"gemPulse 2s ease-in-out infinite"}}><polygon points="90,170 96,158 102,170 96,176" fill="#22c55e"/><polygon points="96,158 99,164 96,170 93,164" fill="#4ade80" opacity="0.6"/></g>
        <g style={{animation:"gemPulse 2s ease-in-out 0.7s infinite"}}><polygon points="300,185 306,173 312,185 306,191" fill="#3b82f6"/><polygon points="306,173 309,179 306,185 303,179" fill="#60a5fa" opacity="0.6"/></g>
        <g style={{animation:"gemPulse 2s ease-in-out 1.4s infinite"}}><polygon points="220,205 227,191 234,205 227,212" fill="#eab308"/><polygon points="227,191 231,198 227,205 223,198" fill="#facc15" opacity="0.6"/></g>
        <g style={{animation:"gemPulse 1.5s ease-in-out 0.3s infinite"}}><polygon points="160,192 163,186 166,192 163,196" fill="#a855f7"/></g>

        {/* === MINER === */}
        <g transform="translate(175, 72)">
          {/* Hard hat */}
          <ellipse cx="0" cy="-18" rx="18" ry="7" fill="#eab308"/><rect x="-16" y="-24" width="32" height="9" rx="5" fill="#facc15"/>
          <circle cx="0" cy="-21" r="3.5" fill="white" style={{animation:"gemPulse 1s infinite"}}/>
          <path d="M0 -21 L-35 15 L35 15 Z" fill="white" opacity="0.04"/>
          {/* Head */}
          <circle cx="0" cy="-6" r="14" fill="#d4a574"/>
          <circle cx="-5" cy="-7" r="2.5" fill="#1e293b"/><circle cx="5" cy="-7" r="2.5" fill="#1e293b"/>
          <circle cx="-4.5" cy="-7.5" r="1" fill="white"/><circle cx="5.5" cy="-7.5" r="1" fill="white"/>
          <path d="M-5 2 Q0 6 5 2" stroke="#5c3a1e" strokeWidth="2" fill="none" strokeLinecap="round"/>
          {/* Body */}
          <rect x="-12" y="8" width="24" height="26" rx="5" fill="#64748b"/>
          <rect x="-13" y="24" width="26" height="5" rx="1.5" fill="#92400e"/><rect x="-3" y="23" width="6" height="7" rx="1.5" fill="#ca8a04"/>
          {/* Legs */}
          <rect x="-11" y="34" width="9" height="18" rx="4" fill="#475569"/><rect x="2" y="34" width="9" height="18" rx="4" fill="#475569"/>
          <rect x="-12" y="49" width="11" height="6" rx="2.5" fill="#44403c"/><rect x="1" y="49" width="11" height="6" rx="2.5" fill="#44403c"/>
          {/* Pickaxe arm */}
          <g style={{transformOrigin:"10px 14px",animation:"pickSwing 0.7s ease-in-out infinite"}}>
            <path d="M12 12 L40 -18" stroke="#d4a574" strokeWidth="6" strokeLinecap="round"/>
            <path d="M34 -12 L62 -48" stroke="#92400e" strokeWidth="4" strokeLinecap="round"/>
            <path d="M54 -54 L70 -42 L62 -46 L66 -36 Z" fill="#94a3b8"/><path d="M54 -54 L62 -46" stroke="#cbd5e1" strokeWidth="1.5"/>
          </g>
          {/* Left arm */}
          <path d="M-12 14 L-24 26" stroke="#d4a574" strokeWidth="6" strokeLinecap="round"/>
        </g>

        {/* Sparks */}
        <g transform="translate(240, 40)">
          <circle cx="0" cy="0" r="4" fill="#facc15" style={{animation:"sparkA 0.7s ease-out infinite"}}/>
          <circle cx="0" cy="0" r="3" fill="#fb923c" style={{animation:"sparkB 0.7s ease-out 0.15s infinite"}}/>
          <circle cx="0" cy="0" r="3.5" fill="#fbbf24" style={{animation:"sparkC 0.7s ease-out 0.3s infinite"}}/>
          <circle cx="0" cy="0" r="3" fill="#f59e0b" style={{animation:"sparkA 0.7s ease-out 0.45s infinite"}}/>
        </g>
        {/* Dust */}
        <g transform="translate(230, 90)">
          <circle cx="0" cy="0" r="9" fill="#a1887f" style={{animation:"dustUp 1s ease-out infinite"}}/>
          <circle cx="-18" cy="5" r="7" fill="#8d6e63" style={{animation:"dustUp 1s ease-out 0.3s infinite"}}/>
          <circle cx="12" cy="3" r="8" fill="#a1887f" style={{animation:"dustUp 1s ease-out 0.6s infinite"}}/>
        </g>
        {/* Cracks */}
        <path d="M228 95 L240 107 L234 118" stroke="#5d4037" strokeWidth="1.5" strokeDasharray="50" style={{animation:"crackDraw 0.7s ease-out infinite"}}/>
        <path d="M234 95 L246 100 L252 112" stroke="#5d4037" strokeWidth="1" strokeDasharray="50" style={{animation:"crackDraw 0.7s ease-out 0.2s infinite"}}/>
        {/* Debris */}
        <g style={{animation:"rockShake 0.35s ease-in-out infinite"}}>
          <rect x="224" y="98" width="6" height="5" rx="1.5" fill="#5d4037" transform="rotate(15 227 100)"/>
          <rect x="244" y="94" width="5" height="4" rx="1" fill="#4e342e" transform="rotate(-10 246 96)"/>
        </g>
        {/* Gem counter */}
        <g transform="translate(30, 25)"><polygon points="0,9 5,0 10,9 5,13" fill="#22c55e" opacity="0.8"/><text x="16" y="11" fontSize="11" fill={t.textDim} fontFamily="monospace">×3</text></g>
        <g transform="translate(30, 48)"><polygon points="0,9 5,0 10,9 5,13" fill="#3b82f6" opacity="0.8"/><text x="16" y="11" fontSize="11" fill={t.textDim} fontFamily="monospace">×7</text></g>
        <g transform="translate(30, 71)"><polygon points="0,9 5,0 10,9 5,13" fill="#eab308" opacity="0.8"/><text x="16" y="11" fontSize="11" fill={t.textDim} fontFamily="monospace">×2</text></g>
        {/* Progress bar */}
        <rect x="80" y="245" width="240" height="7" rx="3.5" fill={t.border}/>
        <rect x="80" y="245" width="0" height="7" rx="3.5" fill={t.accent} style={{animation:"progFill 3s ease-in-out infinite"}}/>
      </svg>
      <div style={{fontSize:17,fontWeight:700,color:t.text,marginTop:14,animation:"txtPulse 1.5s infinite"}}>{msg}</div>
      <div style={{display:"flex",justifyContent:"center",gap:6,marginTop:10}}>
        {[0,1,2,3,4].map(i=><div key={i} style={{width:8,height:8,borderRadius:"50%",background:t.accent,animation:`dotJump 0.6s ease-in-out ${i*0.12}s infinite`}}/>)}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════
// SMALL COMPONENTS
// ══════════════════════════════════════════════════════
function Pill({active,children,onClick,t,s}:{active:boolean;children:React.ReactNode;onClick:()=>void;t:Th;s?:boolean}){
  return <button onClick={onClick} style={{fontSize:s?10:12,fontWeight:active?700:500,padding:s?"4px 9px":"5px 13px",borderRadius:6,cursor:"pointer",border:"none",background:active?t.accent:"transparent",color:active?"#fff":t.textDim,fontFamily:"'Outfit',sans-serif",transition:"all .15s"}}>{children}</button>;
}
const CAT_C:{[k:string]:{c:string;l:string}}={classical:{c:"#3b82f6",l:"Classical"},candlestick:{c:"#f59e0b",l:"Candle"},smb_scalp:{c:"#a855f7",l:"SMB"},quant:{c:"#10b981",l:"Quant"}};
function CatBadge({cat}:{cat:string}){const c=CAT_C[cat]||{c:"#888",l:cat};return <span style={{fontSize:10,fontWeight:700,padding:"2px 6px",borderRadius:4,background:c.c+"18",color:c.c,textTransform:"uppercase",letterSpacing:.3}}>{c.l}</span>;}
function TfBadge({tf,t}:{tf:string;t:Th}){const m=tf.includes("&");return <span style={{fontSize:10,fontWeight:700,padding:"2px 6px",borderRadius:4,background:m?t.accent+"20":t.textDim+"15",color:m?t.accent:t.textDim}}>{m?"★ ":""}{tf}</span>;}
function VerdictBadge({v,t}:{v:any;t:Th}){if(!v||v.verdict==="PENDING")return null;const m:{[k:string]:{bg:string;c:string;i:string}}={CONFIRMED:{bg:t.long+"20",c:t.long,i:"✓"},CAUTION:{bg:t.gold+"20",c:t.gold,i:"⚠"},DENIED:{bg:t.short+"20",c:t.short,i:"✗"}};const c=m[v.verdict]||m.CAUTION;return <span title={v.reasoning} style={{fontSize:10,fontWeight:800,padding:"2px 8px",borderRadius:4,background:c.bg,color:c.c,border:`1px solid ${c.c}30`,cursor:"help"}}>{c.i} {v.verdict}</span>;}
function ScoreCell({score,t}:{score:number;t:Th}){const c=score>=65?t.long:score>=45?t.gold:t.short;return <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:14,fontWeight:800,color:c,padding:"3px 7px",borderRadius:5,background:c+"15",display:"inline-block",minWidth:38,textAlign:"center"}}>{score.toFixed(0)}</span>;}

// ── Correlation Label (shows actual returns) ──
function CorrLabel({corr,t}:{corr:any;t:Th}){
  if(!corr||!corr.label||corr.label==="No data")return null;
  const color=corr.color||t.textDim;
  return(
    <span title={`${corr.symbol}: ${corr.stock_return_pct>=0?"+":""}${corr.stock_return_pct}% vs SPY: ${corr.spy_return_pct>=0?"+":""}${corr.spy_return_pct}% | ${corr.direction_agreement}% dir. agreement`}
      style={{fontSize:10,fontWeight:700,padding:"2px 8px",borderRadius:4,background:color+"18",color,cursor:"help",border:`1px solid ${color}25`,display:"inline-flex",alignItems:"center",gap:4}}>
      <span style={{fontSize:12}}>
        {corr.label==="Relative Strength"||corr.label==="Outperforming"||corr.label==="Holding Up"?"💪":""}
        {corr.label==="Relative Weakness"||corr.label==="Underperforming"?"📉":""}
      </span>
      {corr.label}
      <span style={{opacity:0.6,fontSize:9}}>({corr.spread_pct>=0?"+":""}{corr.spread_pct}%)</span>
    </span>
  );
}

const REG:{[k:string]:{l:string;i:string;ck:keyof Th}}={trending_bull:{l:"TRENDING BULL",i:"▲",ck:"long"},trending_bear:{l:"TRENDING BEAR",i:"▼",ck:"short"},high_volatility:{l:"HIGH VOLATILITY",i:"◆",ck:"gold"},mean_reverting:{l:"MEAN REVERTING",i:"◎",ck:"accent"}};
function RegimePill({r,t}:{r:any;t:Th}){if(!r?.regime)return null;const c=REG[r.regime];if(!c)return null;const co=t[c.ck]as string;return <span style={{fontSize:10,fontWeight:700,padding:"3px 10px",borderRadius:6,background:co+"15",color:co,border:`1px solid ${co}30`}}>{c.i} {c.l}</span>;}

// ══════════════════════════════════════════════════════
// BACKTEST MODAL
// ══════════════════════════════════════════════════════
const PAT_CAT:{[k:string]:string}={"Head & Shoulders":"classical","Inverse H&S":"classical","Double Top":"classical","Double Bottom":"classical","Triple Top":"classical","Triple Bottom":"classical","Ascending Triangle":"classical","Descending Triangle":"classical","Symmetrical Triangle":"classical","Bull Flag":"classical","Bear Flag":"classical","Pennant":"classical","Cup & Handle":"classical","Rectangle":"classical","Rising Wedge":"classical","Falling Wedge":"classical","Bullish Engulfing":"candlestick","Bearish Engulfing":"candlestick","Morning Star":"candlestick","Evening Star":"candlestick","Hammer":"candlestick","Shooting Star":"candlestick","Doji":"candlestick","Dragonfly Doji":"candlestick","Three White Soldiers":"candlestick","Three Black Crows":"candlestick","RubberBand Scalp":"smb_scalp","HitchHiker Scalp":"smb_scalp","ORB 15min":"smb_scalp","ORB 30min":"smb_scalp","Second Chance Scalp":"smb_scalp","BackSide Scalp":"smb_scalp","Fashionably Late":"smb_scalp","Spencer Scalp":"smb_scalp","Gap Give & Go":"smb_scalp","Tidal Wave":"smb_scalp","Breaking News":"smb_scalp","Momentum Breakout":"quant","Vol Compression Breakout":"quant","Mean Reversion":"quant","Trend Pullback":"quant","Gap Fade":"quant","Relative Strength Break":"quant","Range Expansion":"quant","Volume Breakout":"quant","VWAP Reversion":"quant","Donchian Breakout":"quant"};

function BacktestModal({open,onClose,t}:{open:boolean;onClose:()=>void;t:Th}){
  const[data,setData]=useState<any>(null);const[sort,setSort]=useState("edge_score");const[loading,setLoading]=useState(true);
  useEffect(()=>{if(!open)return;setLoading(true);fetch(`${API}/api/backtest/patterns?sort=${sort}`).then(r=>r.json()).then(d=>{setData(d);setLoading(false)}).catch(()=>setLoading(false))},[open,sort]);
  if(!open)return null;const sm=data?.summary||{};const pp=data?.patterns||[];
  return(
    <div style={{position:"fixed",inset:0,zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(0,0,0,0.8)",backdropFilter:"blur(8px)"}} onClick={onClose}>
      <div onClick={e=>e.stopPropagation()} style={{background:t.bgModal,border:`1px solid ${t.border}`,borderRadius:16,width:"95vw",maxWidth:1400,maxHeight:"90vh",overflow:"hidden",display:"flex",flexDirection:"column",boxShadow:"0 30px 60px rgba(0,0,0,0.6)"}}>
        <div style={{padding:"18px 24px",borderBottom:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div><span style={{fontSize:22,fontWeight:800,color:t.text}}>📊 Backtest Results</span>
            {sm.total_signals&&<span style={{fontSize:14,color:t.textDim,marginLeft:14}}>{sm.total_symbols} symbols • {sm.total_signals?.toLocaleString()} signals • {sm.overall_win_rate}% WR</span>}</div>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            {["edge_score","win_rate","profit_factor","expectancy","total_signals"].map(s=>{
              const lb:{[k:string]:string}={edge_score:"Edge",win_rate:"WR%",profit_factor:"PF",expectancy:"Exp",total_signals:"Signals"};
              return <Pill key={s} active={sort===s} onClick={()=>setSort(s)} t={t} s>{lb[s]}</Pill>;})}
            <button onClick={onClose} style={{background:"none",border:"none",color:t.textDim,fontSize:22,cursor:"pointer",marginLeft:12}}>✕</button>
          </div>
        </div>
        <div style={{overflow:"auto",flex:1}}>
          {loading?<MiningAnimation t={t} msg="Loading backtest data..."/>:(
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:14,fontFamily:"'JetBrains Mono',monospace"}}>
              <thead><tr style={{position:"sticky",top:0,background:t.bgModal,borderBottom:`2px solid ${t.borderLight}`,fontSize:11,fontWeight:700,color:t.textMuted,textTransform:"uppercase",letterSpacing:.5}}>
                {["Pattern","Cat","Signals","Win%","PF","Exp/R","Avg W","Avg L","Edge","Grade"].map(h=><th key={h} style={{padding:"12px 10px",textAlign:h==="Pattern"?"left":"right"}}>{h}</th>)}
              </tr></thead>
              <tbody>{pp.map((p:any)=>{
                const gr=p.edge_score>=70&&p.total_signals>=20?"A":p.edge_score>=55&&p.total_signals>=10?"B":p.edge_score>=40&&p.total_signals>=5?"C":p.edge_score>=25?"D":"F";
                const gc:{[k:string]:string}={A:t.long,B:t.accent,C:t.gold,D:t.short,F:t.short};const gc2=gc[gr]||t.textDim;
                return(<tr key={p.name} style={{borderBottom:`1px solid ${t.border}`}}>
                  <td style={{padding:"10px",fontWeight:700,color:t.text,fontFamily:"'Outfit',sans-serif",fontSize:15}}>{p.name}</td>
                  <td style={{textAlign:"right"}}><CatBadge cat={PAT_CAT[p.name]||""}/></td>
                  <td style={{textAlign:"right",color:t.textDim}}>{p.total_signals?.toLocaleString()}</td>
                  <td style={{textAlign:"right",color:p.win_rate>=55?t.long:p.win_rate>=45?t.gold:t.short,fontWeight:700}}>{p.win_rate?.toFixed(1)}%</td>
                  <td style={{textAlign:"right",color:p.profit_factor>=2?t.long:p.profit_factor>=1?t.text:t.short}}>{p.profit_factor?.toFixed(1)}</td>
                  <td style={{textAlign:"right",color:p.expectancy>=0?t.long:t.short}}>{p.expectancy>=0?"+":""}{p.expectancy?.toFixed(3)}</td>
                  <td style={{textAlign:"right",color:t.long}}>{p.avg_win_r?.toFixed(1)}R</td>
                  <td style={{textAlign:"right",color:t.short}}>{p.avg_loss_r?.toFixed(1)}R</td>
                  <td style={{textAlign:"right"}}><span style={{padding:"3px 8px",borderRadius:5,background:gc2+"18",color:gc2,fontWeight:800}}>{p.edge_score?.toFixed(0)}</span></td>
                  <td style={{textAlign:"right"}}><span style={{fontWeight:800,color:gc2,fontSize:16}}>{gr}</span></td>
                </tr>);})}</tbody>
            </table>)}
        </div>
      </div>
    </div>);
}

// ══════════════════════════════════════════════════════
// CHART
// ══════════════════════════════════════════════════════
function TradeChart({setup,onClose,t}:{setup:any;onClose:()=>void;t:Th}){
  const ref=useRef<HTMLDivElement>(null);const[ld,setLd]=useState(true);const[err,setErr]=useState("");
  useEffect(()=>{if(!ref.current)return;let ch:any=null;
    (async()=>{try{const lc=await import("lightweight-charts");const tf=setup.timeframe_detected?.includes("15m")?"15min":"5min";const r=await fetch(`${API}/api/chart/${setup.symbol}?timeframe=${tf}&days_back=5`);const d=await r.json();if(d.error)throw new Error(d.error);if(!d.bars?.length)throw new Error("No data");if(ref.current)ref.current.innerHTML="";
      ch=lc.createChart(ref.current!,{width:ref.current!.clientWidth,height:400,layout:{background:{color:t.chartBg}as any,textColor:t.chartText,fontFamily:"JetBrains Mono,monospace"},grid:{vertLines:{color:t.chartGrid},horzLines:{color:t.chartGrid}},crosshair:{mode:lc.CrosshairMode.Normal},rightPriceScale:{borderColor:t.border},timeScale:{borderColor:t.border,timeVisible:true}});
      let s:any;if((lc as any).CandlestickSeries)s=ch.addSeries((lc as any).CandlestickSeries,{upColor:t.long,downColor:t.short,borderUpColor:t.long,borderDownColor:t.short,wickUpColor:t.long,wickDownColor:t.short});else s=ch.addCandlestickSeries({upColor:t.long,downColor:t.short,borderUpColor:t.long,borderDownColor:t.short,wickUpColor:t.long,wickDownColor:t.short});
      s.setData(d.bars);
      s.createPriceLine({price:setup.entry_price,color:t.accent,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:"ENTRY"});
      s.createPriceLine({price:setup.stop_loss,color:t.short,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:"STOP"});
      s.createPriceLine({price:setup.target_price,color:t.long,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:"TARGET"});
      Object.entries(setup.key_levels||{}).forEach(([n,p])=>{if(typeof p==="number"&&p>0)s.createPriceLine({price:p,color:t.textMuted,lineWidth:1,lineStyle:lc.LineStyle.Dotted,axisLabelVisible:false,title:n})});
      ch.timeScale().fitContent();setLd(false);
      const ro=new ResizeObserver(()=>{if(ref.current&&ch)ch.applyOptions({width:ref.current.clientWidth})});ro.observe(ref.current!);
    }catch(e:any){setErr(e.message);setLd(false)}})();return()=>{if(ch)ch.remove()};},[setup,t]);
  return(<div style={{background:t.bg,border:`1px solid ${t.border}`,borderRadius:12,padding:16,margin:"8px 0"}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
      <span style={{fontSize:16,fontWeight:700,color:t.text}}>{setup.symbol} — {setup.pattern_name} ({setup.timeframe_detected})</span>
      <div style={{display:"flex",gap:14,fontSize:14,fontWeight:700,alignItems:"center"}}>
        <span style={{color:t.accent}}>Entry ${setup.entry_price?.toFixed(2)}</span>
        <span style={{color:t.short}}>Stop ${setup.stop_loss?.toFixed(2)}</span>
        <span style={{color:t.long}}>Target ${setup.target_price?.toFixed(2)}</span>
        <button onClick={onClose} style={{background:"none",border:"none",cursor:"pointer",color:t.textDim,fontSize:18}}>✕</button>
      </div>
    </div>
    {setup.ai_verdict?.reasoning&&setup.ai_verdict.verdict!=="PENDING"&&(
      <div style={{fontSize:13,color:t.textDim,padding:"10px 14px",background:t.bgCard,borderRadius:8,marginBottom:10,border:`1px solid ${t.border}`}}>
        <span style={{fontWeight:700,color:t.text}}>AI Analysis: </span>{setup.ai_verdict.reasoning}
      </div>)}
    {ld&&<div style={{textAlign:"center",padding:30,color:t.textDim}}>Loading chart...</div>}
    {err&&<div style={{textAlign:"center",padding:14,color:t.short,fontSize:13}}>Chart: {err}</div>}
    <div ref={ref} style={{width:"100%",minHeight:ld?0:400,marginTop:8}}/>
  </div>);
}

// ══════════════════════════════════════════════════════
// SETUP ROW — with proper trigger text
// ══════════════════════════════════════════════════════
function SetupRow({s,open,toggle,t,onTrack}:{s:any;open:boolean;toggle:()=>void;t:Th;onTrack:(s:any)=>void}){
  const isL=s.bias==="long";
  // TRIGGER: The price condition, NOT the detection time
  const trigger=isL
    ?`⚡ BUY if ${s.symbol} reaches $${s.entry_price?.toFixed(2)}`
    :`⚡ SHORT if ${s.symbol} breaks below $${s.entry_price?.toFixed(2)}`;

  return(<div>
    <div style={{display:"flex",alignItems:"center",gap:8,padding:"12px 16px",borderBottom:`1px solid ${t.border}`,cursor:"pointer",background:open?t.bgHover:"transparent",transition:"background .15s"}}>
      <div onClick={toggle} style={{display:"flex",alignItems:"center",gap:10,flex:1,minWidth:0}}>
        <div style={{width:70}}>
          <div style={{fontSize:17,fontWeight:800,color:t.text}}>{s.symbol}</div>
          <span style={{fontSize:11,fontWeight:700,padding:"2px 8px",borderRadius:4,background:isL?t.longBg:t.shortBg,color:isL?t.long:t.short}}>{s.bias?.toUpperCase()}</span>
        </div>
        <div style={{flex:1,minWidth:150}}>
          <div style={{display:"flex",alignItems:"center",gap:5,flexWrap:"wrap"}}>
            <span style={{fontSize:15,fontWeight:700,color:t.text}}>{s.pattern_name}</span>
            <CatBadge cat={s.category}/><TfBadge tf={s.timeframe_detected} t={t}/>
            <VerdictBadge v={s.ai_verdict} t={t}/>
            <CorrLabel corr={s.spy_correlation} t={t}/>
          </div>
          {/* THE TRIGGER — the key action line */}
          <div style={{fontSize:13,color:t.accent,fontWeight:700,marginTop:4}}>{trigger}</div>
          <div style={{fontSize:11,color:t.textMuted,marginTop:2}}>
            Target: ${s.target_price?.toFixed(2)} ({s.risk_reward_ratio?.toFixed(1)}R) • Stop: ${s.stop_loss?.toFixed(2)}
          </div>
        </div>
        <div style={{display:"flex",gap:12}}>
          {[{l:"ENTRY",v:s.entry_price,c:t.text},{l:"STOP",v:s.stop_loss,c:t.short},{l:"TARGET",v:s.target_price,c:t.long}].map(({l,v,c})=>(
            <div key={l} style={{textAlign:"right",minWidth:58}}>
              <div style={{fontSize:10,color:t.textMuted,fontWeight:700,letterSpacing:.4}}>{l}</div>
              <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:15,fontWeight:700,color:c}}>${v?.toFixed(2)}</div>
            </div>))}
        </div>
        <ScoreCell score={s.composite_score||0} t={t}/>
        <div style={{width:16,textAlign:"center",fontSize:13,color:t.textDim}}>{open?"▴":"▾"}</div>
      </div>
      <button onClick={e=>{e.stopPropagation();onTrack(s)}} title="Track this trade live" style={{
        background:t.accent+"18",border:`1px solid ${t.accent}35`,borderRadius:6,padding:"6px 12px",
        cursor:"pointer",fontSize:12,fontWeight:700,color:t.accent,whiteSpace:"nowrap",
      }}>+ Track</button>
    </div>
    {open&&<TradeChart setup={s} onClose={toggle} t={t}/>}
  </div>);
}

// ══════════════════════════════════════════════════════
// TRACKED TRADE
// ══════════════════════════════════════════════════════
function TrackedTrade({trade,price,t,onRemove}:{trade:any;price:any;t:Th;onRemove:()=>void}){
  const isL=trade.bias==="long";const entry=trade.entry_price;const stop=trade.stop_loss;const target=trade.target_price;
  const cur=price?.price||0;const risk=Math.abs(entry-stop);
  const pnlR=risk>0?(isL?(cur-entry)/risk:(entry-cur)/risk):0;
  const hitT=isL?cur>=target:cur<=target;const hitS=isL?cur<=stop:cur>=stop;
  const status=hitT?"🎯 TARGET HIT":hitS?"🛑 STOPPED":"⏳ ACTIVE";
  const sc=hitT?t.long:hitS?t.short:t.accent;
  return(<div style={{display:"flex",alignItems:"center",gap:12,padding:"10px 16px",borderBottom:`1px solid ${t.border}`,fontSize:14}}>
    <span style={{fontWeight:800,color:t.text,width:60}}>{trade.symbol}</span>
    <span style={{fontSize:11,padding:"2px 7px",borderRadius:4,background:isL?t.longBg:t.shortBg,color:isL?t.long:t.short,fontWeight:700}}>{trade.bias?.toUpperCase()}</span>
    <span style={{color:t.textDim,flex:1}}>{trade.pattern_name}</span>
    <span style={{fontFamily:"'JetBrains Mono',monospace"}}>
      <span style={{color:t.textDim}}>In: ${entry?.toFixed(2)}</span>
      <span style={{color:t.text,marginLeft:12}}>Now: ${cur?cur.toFixed(2):"..."}</span>
      <span style={{color:pnlR>=0?t.long:t.short,marginLeft:12,fontWeight:800,fontSize:15}}>{pnlR>=0?"+":""}{pnlR.toFixed(2)}R</span>
    </span>
    <span style={{fontSize:12,fontWeight:700,padding:"3px 10px",borderRadius:5,background:sc+"18",color:sc}}>{status}</span>
    <button onClick={onRemove} style={{background:"none",border:"none",color:t.textDim,cursor:"pointer",fontSize:16}}>✕</button>
  </div>);
}

// ══════════════════════════════════════════════════════
// IN-PLAY CARD
// ══════════════════════════════════════════════════════
function InPlayCard({stock,onClick,t}:{stock:any;onClick:()=>void;t:Th}){
  return(<div onClick={onClick} style={{padding:"10px 14px",background:t.bgCard,borderRadius:10,border:`1px solid ${t.border}`,cursor:"pointer",minWidth:130,flex:"0 0 auto",transition:"all .2s"}}
    onMouseEnter={e=>{e.currentTarget.style.borderColor=t.accent;e.currentTarget.style.transform="translateY(-2px)"}}
    onMouseLeave={e=>{e.currentTarget.style.borderColor=t.border;e.currentTarget.style.transform="translateY(0)"}}>
    <span style={{fontSize:16,fontWeight:800,color:t.text}}>{stock.symbol}</span>
    <div style={{fontSize:11,color:t.textDim,marginTop:3}}>{stock.reason?.slice(0,50)}</div>
  </div>);
}

// ══════════════════════════════════════════════════════
// MAIN APP
// ══════════════════════════════════════════════════════
export default function App(){
  const[dark,setDark]=useState(true);const t=dark?DARK:LIGHT;
  const[view,setView]=useState<"opp"|"scan">("opp");
  const[symbol,setSymbol]=useState("AAPL");const[scanSetups,setScanSetups]=useState<any[]>([]);
  const[topSetups,setTopSetups]=useState<any[]>([]);const[inPlay,setInPlay]=useState<any[]>([]);
  const[mktSummary,setMktSummary]=useState("");
  const[loading,setLoading]=useState(false);const[topLoading,setTopLoading]=useState(true);
  const[error,setError]=useState("");const[chartIdx,setChartIdx]=useState<number|null>(null);
  const[fBias,setFBias]=useState("ALL");const[fCat,setFCat]=useState("ALL");const[sortBy,setSortBy]=useState<"score"|"rr">("score");
  const[mode,setMode]=useState<"today"|"active">("today");
  const[regime,setRegime]=useState<any>(null);const[hotStrats,setHotStrats]=useState<any[]>([]);
  const[pc,setPc]=useState(47);const[mktOpen,setMktOpen]=useState(true);
  const[btOpen,setBtOpen]=useState(false);
  const[tracked,setTracked]=useState<any[]>([]);const[prices,setPrices]=useState<any>({});

  useEffect(()=>{
    fetch(`${API}/api/health`).then(r=>r.json()).then(d=>{if(d.patterns)setPc(d.patterns);if(d.market_open!==undefined)setMktOpen(d.market_open)}).catch(()=>{});
    fetch(`${API}/api/regime`).then(r=>r.json()).then(d=>{if(d.regime)setRegime(d)}).catch(()=>{});
    fetch(`${API}/api/hot-strategies?top_n=5`).then(r=>r.json()).then(d=>{if(d.strategies)setHotStrats(d.strategies)}).catch(()=>{});
    setTopLoading(true);
    fetch(`${API}/api/top-opportunities`).then(r=>r.json()).then(d=>{
      if(d.setups)setTopSetups(d.setups);if(d.in_play?.stocks)setInPlay(d.in_play.stocks);
      if(d.market_summary)setMktSummary(d.market_summary);if(d.market_open!==undefined)setMktOpen(d.market_open);
    }).catch(()=>{}).finally(()=>setTopLoading(false));
  },[]);

  useEffect(()=>{if(!tracked.length)return;
    const f=()=>{fetch(`${API}/api/track-prices?symbols=${tracked.map((x:any)=>x.symbol).join(",")}`).then(r=>r.json()).then(d=>{if(d.prices)setPrices(d.prices)}).catch(()=>{})};
    f();const iv=setInterval(f,300000);return()=>clearInterval(iv);
  },[tracked]);

  const handleScan=useCallback(async()=>{
    setLoading(true);setError("");setScanSetups([]);setChartIdx(null);
    try{const r=await fetch(`${API}/api/scan?symbol=${symbol}&mode=${mode}&ai=true`);const d=await r.json();if(d.error)throw new Error(d.error);setScanSetups(d.setups)}catch(e:any){setError(e.message)}finally{setLoading(false)}
  },[symbol,mode]);

  const scanSym=(sym:string)=>{setSymbol(sym);setView("scan");setTimeout(()=>{
    setLoading(true);setError("");setScanSetups([]);setChartIdx(null);
    fetch(`${API}/api/scan?symbol=${sym}&mode=active&ai=true`).then(r=>r.json()).then(d=>{if(!d.error)setScanSetups(d.setups)}).catch(e=>setError(e.message)).finally(()=>setLoading(false));
  },50)};

  const addTrack=(s:any)=>{if(!tracked.find((x:any)=>x.symbol===s.symbol&&x.pattern_name===s.pattern_name))setTracked(p=>[...p,s])};
  const active=view==="opp"?topSetups:scanSetups;
  const filtered=useMemo(()=>{let r=active;if(fBias!=="ALL")r=r.filter((s:any)=>s.bias===fBias.toLowerCase());if(fCat!=="ALL")r=r.filter((s:any)=>s.category===fCat);return[...r].sort((a:any,b:any)=>sortBy==="rr"?b.risk_reward_ratio-a.risk_reward_ratio:(b.composite_score||0)-(a.composite_score||0))},[active,fBias,fCat,sortBy]);

  return(
    <div style={{background:t.bg,minHeight:"100vh",fontFamily:"'Outfit',sans-serif",color:t.text,transition:"background .3s"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.border};border-radius:3px}input:focus,button:focus{outline:none}
      @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}.fade-in{animation:fadeIn .4s ease-out forwards}`}</style>

      {/* ═══ HEADER ═══ */}
      <div style={{padding:"10px 24px",borderBottom:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",alignItems:"center",background:t.bgCard}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <AlphaBeanLogo size={36}/>
          <span style={{fontSize:22,fontWeight:900,letterSpacing:-.5}}>AlphaBean</span>
          <span style={{fontSize:10,color:t.textMuted}}>v3.4</span>
          <div style={{display:"flex",gap:2,marginLeft:14,background:t.border,borderRadius:6,padding:2}}>
            <Pill active={view==="opp"} onClick={()=>setView("opp")} t={t}>Opportunities</Pill>
            <Pill active={view==="scan"} onClick={()=>setView("scan")} t={t}>Scan</Pill>
          </div>
        </div>
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          <RegimePill r={regime} t={t}/>
          {!mktOpen&&<span style={{fontSize:10,padding:"3px 8px",borderRadius:4,background:t.short+"20",color:t.short,fontWeight:700}}>MARKET CLOSED</span>}
          {active.length>0&&<span style={{fontSize:12,color:t.textDim,fontWeight:600}}>{active.filter((s:any)=>s.bias==="long").length}L / {active.filter((s:any)=>s.bias==="short").length}S</span>}
          <button onClick={()=>setBtOpen(true)} style={{fontSize:12,fontWeight:700,padding:"5px 12px",borderRadius:6,border:`1px solid ${t.border}`,background:t.bgCard,color:t.textDim,cursor:"pointer"}}>📊 Backtest</button>
          <button onClick={()=>setDark(!dark)} style={{background:t.border,border:"none",borderRadius:6,padding:"5px 10px",cursor:"pointer",fontSize:14,color:t.text}}>{dark?"☀️":"🌙"}</button>
        </div>
      </div>

      {/* ═══ MAIN ═══ */}
      <div style={{padding:"14px 24px"}}>
        {/* Tracked */}
        {tracked.length>0&&<div className="fade-in" style={{background:t.bgCard,border:`1px solid ${t.accent}30`,borderRadius:12,marginBottom:14,overflow:"hidden"}}>
          <div style={{padding:"8px 16px",borderBottom:`1px solid ${t.border}`,fontSize:13,fontWeight:700,color:t.accent}}>📡 LIVE TRACKING ({tracked.length}) <span style={{color:t.textMuted,fontWeight:500,fontSize:11,marginLeft:10}}>Auto-updates every 5 min</span></div>
          {tracked.map((tr:any,i:number)=><TrackedTrade key={i} trade={tr} price={prices[tr.symbol]} t={t} onRemove={()=>setTracked(p=>p.filter((_:any,idx:number)=>idx!==i))}/>)}
        </div>}

        {/* OPPORTUNITIES */}
        {view==="opp"&&<>
          {mktSummary&&<div className="fade-in" style={{padding:"10px 16px",background:t.bgCard,borderRadius:10,border:`1px solid ${t.border}`,marginBottom:12,fontSize:14,color:t.textDim}}><span style={{fontWeight:700,color:t.text}}>Market: </span>{mktSummary}</div>}
          {inPlay.length>0&&<div className="fade-in" style={{marginBottom:14}}>
            <div style={{fontSize:12,fontWeight:700,color:t.textMuted,marginBottom:6}}>TRENDING ON YAHOO FINANCE</div>
            <div style={{display:"flex",gap:8,overflowX:"auto",paddingBottom:6}}>{inPlay.map((s:any)=><InPlayCard key={s.symbol} stock={s} onClick={()=>scanSym(s.symbol)} t={t}/>)}</div>
          </div>}
          {hotStrats.length>0&&<div className="fade-in" style={{display:"flex",gap:6,marginBottom:12,flexWrap:"wrap",alignItems:"center"}}>
            <span style={{fontSize:12,fontWeight:700,color:t.textMuted}}>🔥 HOT:</span>
            {hotStrats.map((s:any)=><span key={s.name} style={{fontSize:11,padding:"3px 9px",borderRadius:5,background:t.bgCard,border:`1px solid ${t.border}`,color:t.textDim}}><span style={{fontWeight:700,color:t.text}}>{s.name}</span> <span style={{color:t.long}}>{(s.win_rate*100).toFixed(0)}%</span> <span style={{color:t.accent}}>PF {s.profit_factor?.toFixed(1)}</span></span>)}
          </div>}
          {topLoading&&<MiningAnimation t={t} msg="Scanning trending tickers for opportunities..."/>}
        </>}

        {/* SCAN */}
        {view==="scan"&&<div style={{display:"flex",gap:10,alignItems:"end",marginBottom:14,flexWrap:"wrap"}}>
          <div><label style={{fontSize:10,color:t.textMuted,display:"block",marginBottom:3,fontWeight:700}}>SYMBOL</label>
            <input type="text" value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())} onKeyDown={e=>e.key==="Enter"&&handleScan()} placeholder="AAPL" style={{fontSize:16,padding:"8px 12px",borderRadius:8,width:110,border:`1.5px solid ${t.border}`,fontWeight:800,background:t.bgCard,color:t.text,fontFamily:"'Outfit',sans-serif"}}/>
          </div>
          <div style={{display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}><Pill active={mode==="today"} onClick={()=>setMode("today")} t={t} s>Today</Pill><Pill active={mode==="active"} onClick={()=>setMode("active")} t={t} s>Active</Pill></div>
          <button onClick={handleScan} disabled={loading||!symbol} style={{fontSize:14,fontWeight:700,padding:"9px 24px",borderRadius:8,border:"none",background:loading?t.border:t.accent,color:"#fff",cursor:loading?"wait":"pointer"}}>{loading?"Scanning...":"Scan"}</button>
          <span style={{fontSize:12,color:t.textMuted}}>5m+15m • {pc} patterns • AI</span>
        </div>}

        {error&&<div className="fade-in" style={{padding:"10px 14px",borderRadius:8,background:t.shortBg,color:t.short,fontSize:13,marginBottom:12,border:`1px solid ${t.short}30`}}>{error}</div>}
        {loading&&view==="scan"&&<MiningAnimation t={t} msg={`Scanning ${symbol}...`}/>}

        {/* RESULTS */}
        {!loading&&active.length>0&&!(view==="opp"&&topLoading)&&<div className="fade-in">
          <div style={{display:"flex",gap:6,marginBottom:10,flexWrap:"wrap",alignItems:"center"}}>
            <div style={{display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}>{["ALL","LONG","SHORT"].map(b=><Pill key={b} active={fBias===b} onClick={()=>setFBias(b)} t={t} s>{b}</Pill>)}</div>
            <div style={{display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}>{[["ALL","All"],["classical","Classical"],["candlestick","Candle"],["smb_scalp","SMB"],["quant","Quant"]].map(([v,l])=><Pill key={v} active={fCat===v} onClick={()=>setFCat(v)} t={t} s>{l}</Pill>)}</div>
            <div style={{marginLeft:"auto",display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}>{[["score","Score"],["rr","R:R"]].map(([k,l])=><Pill key={k} active={sortBy===k} onClick={()=>setSortBy(k as any)} t={t} s>{l}</Pill>)}</div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:8,padding:"8px 16px",borderBottom:`2px solid ${t.borderLight}`,fontSize:10,fontWeight:700,color:t.textMuted,letterSpacing:.6,textTransform:"uppercase"}}>
            <div style={{width:70}}>Ticker</div><div style={{flex:1}}>Setup / Entry Trigger / Analysis</div>
            <div style={{display:"flex",gap:12}}><div style={{width:58,textAlign:"right"}}>Entry</div><div style={{width:58,textAlign:"right"}}>Stop</div><div style={{width:58,textAlign:"right"}}>Target</div></div>
            <div style={{width:42,textAlign:"center"}}>Score</div><div style={{width:16}}/><div style={{width:75}}/>
          </div>
          {filtered.map((s:any,i:number)=><SetupRow key={`${s.symbol}-${s.pattern_name}-${i}`} s={s} open={chartIdx===i} toggle={()=>setChartIdx(chartIdx===i?null:i)} t={t} onTrack={addTrack}/>)}
          {filtered.length===0&&<div style={{textAlign:"center",padding:30,color:t.textDim,fontSize:14}}>No setups match filters.</div>}
        </div>}

        {!loading&&view==="scan"&&scanSetups.length===0&&!error&&<div style={{textAlign:"center",padding:50,color:t.textDim}}>
          <AlphaBeanLogo size={56}/><div style={{fontSize:18,fontWeight:700,marginTop:14,color:t.text}}>Scan a ticker</div>
          <div style={{fontSize:14,marginTop:6}}>{pc} patterns • 6-factor scoring • AI evaluation</div>
        </div>}

        <div style={{textAlign:"center",marginTop:28,padding:"12px 0",borderTop:`1px solid ${t.border}`}}>
          <span style={{fontSize:11,color:t.textMuted}}>AlphaBean v3.4 — {pc} Detectors — Ollama AI — Yahoo Trending — SPY Correlation</span>
        </div>
      </div>
      <BacktestModal open={btOpen} onClose={()=>setBtOpen(false)} t={t}/>
    </div>);
}