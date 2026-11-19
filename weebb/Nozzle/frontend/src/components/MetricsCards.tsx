import React from 'react';
import { Boxes, Layers, PieChart, Zap } from 'lucide-react';
import { InspectionResult } from '../types';

interface MetricsCardsProps {
  result: InspectionResult | null;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ result }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Total Nozzles</p>
          <h3 className="text-3xl font-black text-sky-400 mt-1">
            {result ? result.total_count : '--'}
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">Detected on Tray</p>
        </div>
        <div className="p-3 bg-sky-500/10 rounded-xl text-sky-400">
          <Boxes className="w-6 h-6" />
        </div>
      </div>

      <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Unique Types</p>
          <h3 className="text-3xl font-black text-purple-400 mt-1">
            {result ? result.unique_types : '--'}
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">Distinct Classes</p>
        </div>
        <div className="p-3 bg-purple-500/10 rounded-xl text-purple-400">
          <Layers className="w-6 h-6" />
        </div>
      </div>

      <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Tray Capacity</p>
          <h3 className="text-3xl font-black text-emerald-400 mt-1">
            {result ? `${result.occupancy_rate}%` : '--%'}
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">Based on 77 Slots</p>
        </div>
        <div className="p-3 bg-emerald-500/10 rounded-xl text-emerald-400">
          <PieChart className="w-6 h-6" />
        </div>
      </div>

      <div className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Inference Speed</p>
          <h3 className="text-3xl font-black text-amber-400 mt-1">
            {result ? `${result.latency_ms} ms` : '-- ms'}
          </h3>
          <p className="text-[11px] text-slate-500 mt-0.5">FastAPI + GPU</p>
        </div>
        <div className="p-3 bg-amber-500/10 rounded-xl text-amber-400">
          <Zap className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};
