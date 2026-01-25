/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'drone-dark': '#0a1628',
        'drone-darker': '#060d18',
        'drone-card': '#0f1d32',
        'drone-border': '#1a3050',
        'drone-accent': '#00d4aa',
        'drone-red': '#ff4757',
        'drone-green': '#2ed573',
        'drone-blue': '#3498db',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
      }
    },
  },
  plugins: [],
}
