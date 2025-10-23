"use client"

export default function StatsCards({ totalPatients = 128, todayConsults = 7, pendingReports = 3 }) {
  const cards = [
    {
      label: "Total Patients",
      value: totalPatients,
      icon: UsersIcon,
    },
    {
      label: "Today’s Consultations",
      value: todayConsults,
      icon: CalendarIcon,
    },
    {
      label: "Pending Reports",
      value: pendingReports,
      icon: ReportIcon,
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {cards.map(({ label, value, icon: Icon }, index) => (
        <div key={index} className="card p-4 flex items-center gap-4">
          <div className="h-10 w-10 rounded-lg bg-[color:var(--color-soft)] text-[color:var(--color-brand)] flex items-center justify-center">
            <Icon />
          </div>
          <div className="min-w-0">
            <p className="text-sm text-muted-foreground">{label}</p>
            <p className="text-xl font-semibold">{String(value)}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

function UsersIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
      <path d="M16 11a4 4 0 10-8 0 4 4 0 008 0ZM3 20a7 7 0 0114 0v1H3v-1Z" />
      <path
        d="M17.5 9.5a3.5 3.5 0 110-7 3.5 3.5 0 010 7ZM18 14a6 6 0 016 6v1h-4v-1a2 2 0 00-2-2h-1v-4z"
        opacity="0.3"
      />
    </svg>
  )
}
function CalendarIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
      <path d="M7 2a1 1 0 011 1v1h8V3a1 1 0 112 0v1h1a2 2 0 012 2v3H2V6a2 2 0 012-2h1V3a1 1 0 012 0v1z" />
      <path d="M2 10h20v9a2 2 0 01-2 2H4a2 2 0 01-2-2v-9zm5 3h3v3H7v-3z" />
    </svg>
  )
}
function ReportIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
      <path d="M6 2h8l6 6v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4a2 2 0 012-2z" />
      <path d="M14 2v6h6" opacity="0.3" />
      <path d="M8 12h8v2H8zm0 4h6v2H8z" />
    </svg>
  )
}
