"use client"

import { useState } from "react"
import PrescriptionForm from "@/components/prescription-form"
import PrescriptionPreview from "@/components/prescription-preview"

export default function PrescriptionsPage() {
  const [data, setData] = useState(null)

  return (
    <main className="mx-auto max-w-6xl p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Create Prescription</h1>
        <p className="text-sm text-muted-foreground">
          Enter or auto-fill details, review warnings, generate a preview, and export or send to the patient.
        </p>
      </header>

      <div className="grid gap-6 md:grid-cols-2">
        <section className="rounded-lg border border-border bg-card p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-base font-semibold text-foreground">Prescription Details</h2>
            <span className="rounded-md bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
              Professional Mode
            </span>
          </div>
          <PrescriptionForm onGenerate={setData} />
        </section>

        <section className="rounded-lg border border-border bg-card p-6">
          <PrescriptionPreview data={data} />
        </section>
      </div>
    </main>
  )
}
