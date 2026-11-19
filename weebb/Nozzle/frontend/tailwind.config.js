/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  safelist: [
    // KPI card dynamic colors
    'text-sky-400', 'text-purple-400', 'text-emerald-400', 'text-amber-400',
    'bg-sky-500/10', 'bg-purple-500/10', 'bg-emerald-500/10', 'bg-amber-500/10',
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#060911',
          900: '#0b0f19',
          800: '#111827',
          700: '#1f2937',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
