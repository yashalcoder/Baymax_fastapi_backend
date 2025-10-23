function Benefit({ title, desc }) {
  return (
    <li className="flex items-start gap-3">
      <div className="mt-1 h-5 w-5 rounded-full bg-brand/10 flex items-center justify-center shrink-0">
        <span className="text-brand text-sm">✓</span>
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="text-muted-foreground text-sm">{desc}</div>
      </div>
    </li>
  )
}

function Mockup() {
  return (
    <div className="card p-4">
      <div className="h-3 rounded-md bg-brand" />
      <div className="space-y-3 mt-4">
        <div className="card p-4 bg-secondary">
          <div className="text-sm text-muted-foreground">Today's Appointments</div>
          <div className="text-xl font-semibold">24 Patients</div>
        </div>
        <div className="rounded-lg border border-green-200 bg-green-50 p-4">
          <div className="text-sm font-medium">AI Suggestion</div>
          <div className="text-sm text-muted-foreground">
            Consider viral upper respiratory infection based on symptoms: fever, cough, fatigue.
          </div>
        </div>
        <div className="card p-4">
          <div className="font-medium">Recent Transcriptions</div>
          <div className="text-sm text-muted-foreground">"بخار تین دن سے ہے..."</div>
        </div>
      </div>
    </div>
  )
}

export default function Providers() {
  return (
    <div className="grid md:grid-cols-2 gap-10 items-center">
      <div className="order-2 md:order-1">
        <div className="inline-flex items-center text-brand bg-brand/10 px-3 py-1 rounded-full text-sm mb-4">
          For Healthcare Providers
        </div>
        <h3 className="section-title">
          Empower Your Practice with <span className="text-brand">AI Assistance</span>
        </h3>
        <p className="text-muted-foreground mt-4 md:text-lg">
          Focus on patient care while BayMax+ handles documentation and decision support.
        </p>
        <ul className="mt-6 space-y-4">
          <Benefit
            title="Save Time"
            desc="Reduce documentation time by up to 70% with automated transcription and record keeping."
          />
          <Benefit
            title="Better Communication"
            desc="Break language barriers with bilingual support in Urdu and English."
          />
          <Benefit title="Reduce Errors" desc="AI-powered safety checks for allergies and drug interactions." />
          <Benefit title="Smart Assistance" desc="Get diagnosis suggestions based on symptoms and history." />
        </ul>
      </div>
      <div className="order-1 md:order-2">
        <Mockup />
      </div>
    </div>
  )
}
