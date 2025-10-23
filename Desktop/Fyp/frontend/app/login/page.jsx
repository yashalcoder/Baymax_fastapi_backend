"use client";
import { useState } from "react";
import { Stethoscope } from "lucide-react";
import { useRouter } from "next/navigation";
export default function LoginPage() {
  const [role, setRole] = useState("doctor");
  const router = useRouter();
  return (
    <main className="min-h-[88vh] bg-soft flex items-center justify-center px-4 py-16">
      <div className="w-full max-w-xl card p-6 md:p-8">
        <div className="flex flex-col items-center text-center gap-3">
          <a href="#" className="flex items-center gap-2 font-semibold">
            <Stethoscope className="w-10 h-10 text-white bg-hero-gradient p-1 rounded-md" />
            <div>
              <span>BayMax+</span>
              <p className="text-xs text-gray-900 b">Healthcare Reimagined</p>
            </div>
          </a>
          <h1 className="text-3xl font-semibold">Welcome Back</h1>
          <p className="text-muted-foreground">
            Sign in to access your healthcare dashboard
          </p>
        </div>

        <div
          className="mt-5 grid grid-cols-2 gap-2 p-1 rounded-lg border bg-background"
          role="tablist"
          aria-label="Select role"
        >
          <button
            type="button"
            role="tab"
            aria-selected={role === "patient"}
            onClick={() => setRole("patient")}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              role === "patient" ? "bg-secondary" : ""
            }`}
          >
            Patient
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={role === "doctor"}
            onClick={() => setRole("doctor")}
            className={`px-4 py-2 rounded-md text-sm font-medium ${
              role === "doctor" ? "bg-secondary" : ""
            }`}
          >
            Doctor
          </button>
        </div>

        <form
          className="mt-6 space-y-4"
          onSubmit={(e) => e.preventDefault()}
          aria-label="Sign in form"
        >
          <div>
            <label htmlFor="email" className="block text-sm font-medium mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-brand"
              placeholder="your@email.com"
            />
          </div>
          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium mb-1"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-brand"
              placeholder="Enter your password"
            />
          </div>
          <button
            type="submit"
            className="w-full hover:cursor-pointer px-4 py-3 rounded-md bg-hero-gradient text-white font-medium hover:opacity-90"
            onClick={() => {
              router.push("/doctor");
            }}
          >
            Sign In
          </button>
        </form>

        <p className="text-center text-sm text-muted-foreground mt-4">
          Don’t have an account?{" "}
          <a className="text-brand hover:underline" href="/signup">
            Sign up
          </a>
        </p>
      </div>
    </main>
  );
}
