function Badge({ title, subtitle }) {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-background border shadow-sm">
      <div className="h-8 w-8 rounded-full bg-brand/10 flex items-center justify-center">
        <span className="text-brand text-lg">•</span>
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="text-xs text-muted-foreground">{subtitle}</div>
      </div>
    </div>
  )
}

export default function BadgeRow() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Badge title="Bilingual" subtitle="Urdu & English" />
      <Badge title="AI‑Powered" subtitle="Smart Diagnosis" />
      <Badge title="Secure" subtitle="HIPAA Compliant" />
    </div>
  )
}
