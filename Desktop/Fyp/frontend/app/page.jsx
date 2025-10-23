import Header from "../components/header";
import Hero from "../components/hero";
import BadgeRow from "../components/badge-row";
import Features from "../components/features";
import Providers from "../components/providers";
import Patients from "../components/patients";
import HowItWorks from "../components/how-it-works";
import CTA from "../components/cta";
import Footer from "../components/footer";

export default function Page() {
  return (
    <main>
      <Header />
      <Hero />

      <section id="features">
        <div className="max-w-6xl mx-auto px-4 py-20">
          <div className="text-center space-y-4 mb-12">
            <h2 className="section-title ">
              Powerful Features for{" "}
              <span className="text-brand">Better Healthcare</span>
            </h2>
            <p className="section-subtitle mx-auto">
              Everything you need to provide world-class healthcare in Pakistan
            </p>
          </div>
          <Features />
        </div>
      </section>

      <section id="providers" className="bg-background">
        <div className="max-w-6xl mx-auto px-4 py-20">
          <Providers />
        </div>
      </section>

      <section id="patients" className="bg-background">
        <div className="max-w-6xl mx-auto px-4 py-20">
          <Patients />
        </div>
      </section>

      <section id="how-it-works" className="bg-soft">
        <div className="max-w-6xl mx-auto px-4 py-20">
          <HowItWorks />
        </div>
      </section>

      <section className="bg-background">
        <div className="max-w-6xl mx-auto px-4 py-20">
          <CTA />
        </div>
      </section>

      <Footer />
    </main>
  );
}
