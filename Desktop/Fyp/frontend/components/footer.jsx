import { Stethoscope } from "lucide-react";
export default function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="max-w-6xl mx-auto px-4 py-12 grid md:grid-cols-3 gap-8">
        <div>
          <a href="#" className="flex items-center gap-2 font-semibold">
            <Stethoscope className="w-10 h-10 text-white bg-hero-gradient p-1 rounded-md" />
            <div>
              <span>BayMax+</span>
              <p className="text-xs text-gray-900 b">Healthcare Reimagined</p>
            </div>
          </a>
          <p className="text-sm text-muted-foreground mt-3">
            Transforming healthcare in Pakistan with bilingual AI assistance,
            automated record keeping, and affordable medicine access.
          </p>
          <div className="flex gap-4 text-xs text-muted-foreground mt-3">
            <span>DRAP Compliant</span>
            <span>HIPAA Certified</span>
            <span>Bilingual Support</span>
          </div>
        </div>
        <div>
          <div className="font-semibold mb-3">Quick Links</div>
          <ul className="space-y-2 text-sm">
            <li>
              <a className="hover:text-brand" href="#features">
                Features
              </a>
            </li>
            <li>
              <a className="hover:text-brand" href="#how-it-works">
                How It Works
              </a>
            </li>
            <li>
              <a className="hover:text-brand" href="#providers">
                For Doctors
              </a>
            </li>
            <li>
              <a className="hover:text-brand" href="#patients">
                For Patients
              </a>
            </li>
          </ul>
        </div>
        <div>
          <div className="font-semibold mb-3">Contact</div>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>FAST‑NUCES Chiniot</li>
            <li>Pakistan</li>
            <li>support@baymaxplus.com</li>
          </ul>
        </div>
      </div>
      <div className="border-t border-border">
        <div className="max-w-6xl mx-auto px-4 py-6 text-xs text-muted-foreground">
          © 2025 BayMax+. Final Year Project — FAST‑NUCES Chiniot
        </div>
      </div>
    </footer>
  );
}
