"use client"

import { useMemo, useState } from "react"

export default function PrescriptionForm({ onGenerate }) {
  const [patientName, setPatientName] = useState("")
  const [symptoms, setSymptoms] = useState("")
  const [medicines, setMedicines] = useState("")
  const [dosage, setDosage] = useState("")
  const [duration, setDuration] = useState("")
  const [allergies, setAllergies] = useState("")

  const allergyKeywords = [
    "penicillin",
    "amoxicillin",
    "aspirin",
    "ibuprofen",
    "metronidazole",
    "cephalosporin",
    "sulfa",
  ]
  const interactionPairs = [
    { a: "warfarin", b: "nsaid", message: "Warfarin + NSAIDs may increase bleeding risk." },
    { a: "warfarin", b: "aspirin", message: "Warfarin + Aspirin may increase bleeding risk." },
    { a: "metformin", b: "contrast", message: "Metformin + Contrast dyes may increase lactic acidosis risk." },
    { a: "ace inhibitor", b: "spironolactone", message: "ACE inhibitors + Spironolactone may raise potassium levels." },
    { a: "statin", b: "clarithromycin", message: "Statins + Clarithromycin may increase myopathy risk." },
  ]

  const textIncludes = (haystack, needle) => haystack.toLowerCase().includes(needle.toLowerCase())

  const warnings = useMemo(() => {
    const w = []
    if (allergies.trim() && medicines.trim()) {
      const aLower = allergies.toLowerCase()
      const mLower = medicines.toLowerCase()
      allergyKeywords.forEach((kw) => {
        if (aLower.includes(kw) && mLower.includes(kw)) {
          w.push(`Allergy alert: Patient allergic to ${kw}, but it appears in medicines.`)
        }
      })
    }
    if (medicines.trim()) {
      const m = medicines.toLowerCase()
      interactionPairs.forEach(({ a, b, message }) => {
        if (textIncludes(m, a) && textIncludes(m, b)) {
          w.push(`Interaction warning: ${message}`)
        }
      })
    }
    return w
  }, [allergies, medicines])

  function handleAutofill() {
    setPatientName("Ahmed Khan")
    setSymptoms("Fever, cough, mild sore throat for 3 days")
    setMedicines("Paracetamol 500mg\nAzithromycin 250mg\nCough Syrup")
    setDosage("Paracetamol: 1 tablet every 8 hours\nAzithromycin: 1 tablet daily\nCough Syrup: 10ml at night")
    setDuration("5 days")
    setAllergies("Penicillin")
  }

  function handleSubmit(e) {
    e.preventDefault()
    const payload = {
      patientName: patientName.trim(),
      symptoms: symptoms.trim(),
      medicines: medicines.trim(),
      dosage: dosage.trim(),
      duration: duration.trim(),
      allergies: allergies.trim(),
      warnings,
      createdAt: new Date().toISOString(),
    }
    onGenerate(payload)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="patientName" className="block text-sm font-medium text-foreground">
            Patient Name
          </label>
          <input
            id="patientName"
            name="patientName"
            value={patientName}
            onChange={(e) => setPatientName(e.target.value)}
            placeholder="Enter patient full name"
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
        </div>
        <div>
          <label htmlFor="allergies" className="block text-sm font-medium text-foreground">
            Known Allergies
          </label>
          <input
            id="allergies"
            name="allergies"
            value={allergies}
            onChange={(e) => setAllergies(e.target.value)}
            placeholder="e.g., Penicillin, Aspirin"
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
        </div>
      </div>

      <div>
        <label htmlFor="symptoms" className="block text-sm font-medium text-foreground">
          Symptoms
        </label>
        <textarea
          id="symptoms"
          name="symptoms"
          rows={3}
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          placeholder="Describe symptoms in your own words (Urdu/English/Punjabi supported)"
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="medicines" className="block text-sm font-medium text-foreground">
            Medicines
          </label>
          <textarea
            id="medicines"
            name="medicines"
            rows={6}
            value={medicines}
            onChange={(e) => setMedicines(e.target.value)}
            placeholder={"One item per line, e.g.\nParacetamol 500mg\nAzithromycin 250mg"}
            className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
        </div>
        <div className="grid gap-4">
          <div>
            <label htmlFor="dosage" className="block text-sm font-medium text-foreground">
              Dosage Instructions
            </label>
            <textarea
              id="dosage"
              name="dosage"
              rows={3}
              value={dosage}
              onChange={(e) => setDosage(e.target.value)}
              placeholder={"Paracetamol: 1 tablet every 8 hours\nAzithromycin: 1 tablet daily"}
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
            />
          </div>
          <div>
            <label htmlFor="duration" className="block text-sm font-medium text-foreground">
              Duration
            </label>
            <input
              id="duration"
              name="duration"
              value={duration}
              onChange={(e) => setDuration(e.target.value)}
              placeholder="e.g., 5 days"
              className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary"
            />
          </div>
        </div>
      </div>

      <div aria-live="polite" className="rounded-lg border border-border bg-card p-3 text-sm">
        <div className="mb-2 flex items-center gap-2">
          <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-primary/10 text-primary">
            {/* Alert icon */}
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
              <path d="M12 2c.6 0 1.1.3 1.4.8l8.1 14a1.6 1.6 0 0 1-1.4 2.4H3.9A1.6 1.6 0 0 1 2.5 16.8l8.1-14c.3-.5.8-.8 1.4-.8zm0 6a1 1 0 0 0-1 1v4a1 1 0 1 0 2 0V9a1 1 0 0 0-1-1zm0 8.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5z" />
            </svg>
          </span>
          <p className="font-medium">Allergy & Interaction Warnings</p>
        </div>
        {warnings.length === 0 ? (
          <p className="text-muted-foreground">No warnings detected based on current entries.</p>
        ) : (
          <ul className="list-disc pl-5 text-foreground">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleAutofill}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm font-medium text-foreground hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-primary"
        >
          {/* Wand icon */}
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            <path d="M6 2h2v3H6V2zm10 0h2v3h-2V2zM3 8h3v2H3V8zm15 0h3v2h-3V8zM6 19h2v3H6v-3zm10 0h2v3h-2v-3zM8 10h8v4H8v-4z" />
          </svg>
          Auto-fill Example
        </button>

        <button
          type="submit"
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-sm hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary"
        >
          {/* Generate icon */}
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor" aria-hidden="true">
            <path d="M12 2l3 7h7l-5.5 4.2L18 21l-6-4-6 4 1.5-7.8L2 9h7z" />
          </svg>
          Generate Prescription
        </button>
      </div>
    </form>
  )
}
