export default function CTA() {
  return (
    <div
      id="cta"
      className="card p-8 md:p-10 flex flex-col md:flex-row items-center md:items-center justify-between gap-6"
    >
      <div>
        <div className="text-sm text-muted-foreground">Secure & Private</div>
        <h4 className="text-2xl md:text-3xl font-semibold mt-1">
          Create Your Account
        </h4>
        <p className="text-muted-foreground mt-2">
          Your data is encrypted and protected with bank‑level security.
        </p>
      </div>
      <a
        href="#"
        className="px-5 py-3 rounded-md bg-hero-gradient text-white font-medium hover:opacity-90"
      >
        Get Started
      </a>
    </div>
  );
}
