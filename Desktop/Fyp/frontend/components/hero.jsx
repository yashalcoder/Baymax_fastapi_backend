import { Stethoscope } from "lucide-react";

export default function Hero() {
  return (
    <section className="bg-soft border-b  border-border">
      <div className="max-w-6xl mx-auto px-4 py-20 md:py-28">
        <div className="flex flex-col items-center text-center gap-6">
          <div className="text-xs flex gap-2 text-blue-700 md:text-sm px-3 py-1 rounded-full border bg-background shadow shadow-md font-semibold">
            <Stethoscope size={20} /> DRAP Compliant • Secure • Bilingual
          </div>
          <h1 className="section-title md:text-6xl text-black">Baymax+</h1>
          <h2 className="md:text-5xl text-brand font-bold">
            Smarter, Bilingual Healthcare
          </h2>
          <p className="text-gray-500 max-w-2xl md:text-xl">
            Empowering doctors and patients in Pakistan with AI-powered
            bilingual transcription, smart diagnosis suggestions, and affordable
            medicine alternatives.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 mt-2">
            <a
              href="#cta"
              className="px-5 py-3 rounded-md bg-hero-gradient text-white font-medium hover:opacity-90"
            >
              Get Started Free →
            </a>
            <a
              href="#how-it-works"
              className="px-5 py-3 rounded-md border bg-background hover:bg-secondary font-medium"
            >
              Watch Demo
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
