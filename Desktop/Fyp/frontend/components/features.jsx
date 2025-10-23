import {
  Languages,
  FileText,
  BrainCircuit,
  Pill,
  ScanSearch,
  UserSquare2,
  ShieldCheck,
  Mic,
} from "lucide-react";
const items = [
  {
    title: "Bilingual Transcription",
    desc: "Real-time transcription of doctor‑patient conversations in both Urdu and English, ensuring nothing gets lost in translation.",
    icon: Languages,
  },
  {
    title: "Digital Medical Records",
    desc: "Secure storage of patient history, symptoms, allergies, and treatments in one centralized, easily accessible system.",
    icon: FileText,
  },
  {
    title: "AI Diagnosis Support",
    desc: "Intelligent suggestions for possible diagnoses based on symptoms and medical history to assist doctors in decision-making.",
    icon: BrainCircuit,
  },
  {
    title: "Smart Prescriptions",
    desc: "Automated prescription generation with interaction checks, allergy alerts, and DRAP compliance.",
    icon: Pill,
  },
  {
    title: "Medicine Price Comparison",
    desc: "Compare medicine prices and availability from local pharmacies to help patients find affordable options.",
    icon: ScanSearch,
  },
  {
    title: "Patient Portal",
    desc: "Patients can access their records, prescriptions, and treatment history anytime, anywhere.",
    icon: UserSquare2,
  },
  {
    title: "Safety Checks",
    desc: "Comprehensive allergy screening and harmful drug interaction detection before finalizing any prescription.",
    icon: ShieldCheck,
  },
  {
    title: "Voice Recognition",
    desc: "Accurate speech‑to‑text tuned for medical terms in both Urdu and English.",
    icon: Mic,
  },
];

function FeatureCard({ title, desc, icon: Icon }) {
  return (
    <div className="card bg-white shadow shadow-md hover:shadow-blue-200 hover:scale-105 transition-all duration-300 p-6">
      <div className="h-10 w-10 rounded-lg bg-brand/10 flex items-center justify-center mb-4">
        <Icon className="text-brand w-6 h-6" />{" "}
        {/* ✅ use <Icon /> not {icon} */}
      </div>
      <h3 className="font-semibold text-lg mb-2">{title}</h3>
      <p className="text-muted-foreground text-sm leading-relaxed">{desc}</p>
    </div>
  );
}

export default function Features() {
  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {items.map((it) => (
        <FeatureCard
          key={it.title}
          title={it.title}
          desc={it.desc}
          icon={it.icon}
        />
      ))}
    </div>
  );
}
