"use client"
import { useState } from "react"

export default function SignupPage() {
  const [role, setRole] = useState("patient")

  return (
    <main className="min-h-[88vh] bg-soft flex items-center justify-center px-4 py-16">
      <div className="w-full max-w-xl card p-6 md:p-8">
        <div className="flex flex-col items-center text-center gap-3">
          <img src="/images/auth-hero.png" alt="BayMax+ preview" className="h-16 w-auto object-contain" />
          <h1 className="text-3xl font-semibold">Create Account</h1>
          <p className="text-muted-foreground">Join BayMax+ for better healthcare management</p>
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
            className={`px-4 py-2 rounded-md text-sm font-medium ${role === "patient" ? "bg-secondary" : ""}`}
          >
            Patient
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={role === "doctor"}
            onClick={() => setRole("doctor")}
            className={`px-4 py-2 rounded-md text-sm font-medium ${role === "doctor" ? "bg-secondary" : ""}`}
          >
            Doctor
          </button>
        </div>

        <form className="mt-6 space-y-4" onSubmit={(e) => e.preventDefault()} aria-label="Create account form">
          <div>
            <label htmlFor="name" className="block text-sm font-medium mb-1">
              Full Name
            </label>
            <input
              id="name"
              type="text"
              required
              className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-brand"
              placeholder="Enter your name"
            />
          </div>
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
            <label htmlFor="password" className="block text-sm font-medium mb-1">
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-brand"
              placeholder="Create a password"
            />
          </div>
          <div>
            <label htmlFor="confirm" className="block text-sm font-medium mb-1">
              Confirm Password
            </label>
            <input
              id="confirm"
              type="password"
              required
              className="w-full rounded-md border bg-background px-3 py-2 outline-none focus:ring-2 focus:ring-brand"
              placeholder="Confirm your password"
            />
          </div>
          <button
            type="submit"
            className="w-full px-4 py-3 rounded-md bg-brand text-white font-medium hover:opacity-90"
          >
            Create Account
          </button>
        </form>
      </div>
    </main>
  )
}
