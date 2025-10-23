const steps = [
  {
    title: "Conversation Capture",
    desc: "Doctor‑patient conversations are transcribed in real time in both Urdu and English using advanced AI.",
  },
  {
    title: "AI Analysis",
    desc: "We extract symptoms, allergies, and history, then suggest possible diagnoses.",
  },
  {
    title: "Smart Prescription",
    desc: "Generate prescriptions with safety checks, interactions, and DRAP compliance.",
  },
  {
    title: "Patient Empowerment",
    desc: "Patients access records and compare medicine prices for the best value.",
  },
];

export default function HowItWorks() {
  return (
    <div>
      <div className="text-center space-y-4 mb-12">
        <h3 className="section-title">
          How <span className="text-brand">BayMax+</span> Works
        </h3>
        <p className="section-subtitle mx-auto">
          A simple, four‑step process that transforms healthcare delivery
        </p>
      </div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {steps.map((s, i) => (
          <div
            className="card p-6 shadow shadow-md hover:scale-105 transition-all duration-300 hover:shadow-blue-300"
            key={s.title}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="h-10 w-10 rounded-full bg-brand/10 flex items-center justify-center">
                <span className="text-brand font-semibold">{i + 1}</span>
              </div>
            </div>
            <div className="font-semibold text-lg">{s.title}</div>
            <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
              {s.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
