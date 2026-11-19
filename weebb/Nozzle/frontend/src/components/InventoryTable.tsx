import React from 'react';
import { ClipboardList, Download, CheckCircle2 } from 'lucide-react';
import { InspectionResult } from '../types';

interface InventoryTableProps {
  result: InspectionResult | null;
}

export const InventoryTable: React.FC<InventoryTableProps> = ({ result }) => {
  const exportToCSV = () => {
    if (!result || !result.breakdown || result.breakdown.length === 0) return;

    let csv = 'Nozzle Type,Quantity,Tray Share (%),Quality Status\n';
    result.breakdown.forEach((item) => {
      csv += `"${item.type}",${item.count},"${item.percentage}%","Passed QC"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `nozzle_tray_bom_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const hasData = result && result.breakdown && result.breakdown.length > 0;

  return (
    <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <ClipboardList className="w-4 h-4 text-sky-400" /> Automated Bill of Materials (BOM) Inventory
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Categorized counts and tray allocation percentage</p>
        </div>
        <button
          onClick={exportToCSV}
          disabled={!hasData}
          className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-2 transition disabled:opacity-40 disabled:cursor-not-allowed shadow-sm"
        >
          <Download className="w-3.5 h-3.5 text-sky-400" /> Export CSV Report
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
              <th className="py-3 px-4">Nozzle Class / Type</th>
              <th className="py-3 px-4">Quantity</th>
              <th className="py-3 px-4">Tray Share (%)</th>
              <th className="py-3 px-4">Inspection QC Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-300">
            {hasData ? (
              result.breakdown.map((item, idx) => (
                <tr key={idx} className="hover:bg-slate-800/20 transition">
                  <td className="py-3 px-4 font-medium text-white">{item.type}</td>
                  <td className="py-3 px-4 font-mono font-bold text-sky-400">{item.count}</td>
                  <td className="py-3 px-4 font-mono">{item.percentage}%</td>
                  <td className="py-3 px-4">
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      <CheckCircle2 className="w-3 h-3" /> Passed QC
                    </span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} className="py-6 text-center text-slate-500 italic">
                  {result
                    ? 'No nozzles detected with current confidence threshold.'
                    : 'Use the live camera or upload a tray photo to generate inventory.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
