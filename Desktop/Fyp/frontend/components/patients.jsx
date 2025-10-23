function Item({ title, desc }) {
  return (
    <li className="flex items-start gap-3">
      <div className="mt-1 h-5 w-5 rounded-full bg-brand/10 flex items-center justify-center shrink-0">
        <span className="text-brand text-sm">✓</span>
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="text-sm text-muted-foreground">{desc}</div>
      </div>
    </li>
  )
}

function PatientMock() {
  return (
    <div className="card p-4">
      <div className="h-3 rounded-md bg-brand" />
      <div className="p-4 card mt-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-brand/10" />
          <div>
            <div className="font-semibold">Ahmed Khan</div>
            <div className="text-xs text-muted-foreground">Patient ID: #12345</div>
          </div>
        </div>
      </div>
      <div className="card p-4 mt-3">
        <div className="text-sm font-medium">Recent Prescriptions</div>
        <div className="text-sm text-muted-foreground mt-1">Paracetamol 500mg – 3x daily</div>
      </div>
      <div className="rounded-lg border bg-emerald-50 p-4 mt-3">
        <div className="font-medium">Medicine Price Comparison</div>
        <div className="text-sm text-muted-foreground mt-1">Dawaai.pk — Rs. 45 • Sehat.com.pk — Rs. 52</div>
      </div>
      <div className="rounded-lg border bg-sky-50 p-4 mt-3">
        <div className="font-medium">Allergy Alert</div>
        <div className="text-sm text-muted-foreground">Penicillin — Recorded on file</div>
      </div>
    </div>
  )
}

export default function Patients() {
  return (
    <div className="grid md:grid-cols-2 gap-10 items-center">
      <div className="order-2">
        <h3 className="section-title">
          Take Control of Your <span className="text-brand">Health</span> Journey
        </h3>
        <p className="text-muted-foreground mt-4 md:text-lg">
          Access your complete medical history, view prescriptions, and find the most affordable medicines — all in one
          secure place.
        </p>
        <ul className="mt-6 space-y-4">
          <Item title="Your Records, Your Control" desc="Access your complete medical history anytime, anywhere." />
          <Item title="Complete Medical History" desc="Never lose track of past treatments, allergies, or diagnoses." />
          <Item title="Save Money" desc="Compare medicine prices and find affordable options nearby." />
          <Item title="Secure & Private" desc="Bank‑level encryption keeps your data protected." />
        </ul>
      </div>
      <div className="order-1">
        <PatientMock />
      </div>
    </div>
  )
}
