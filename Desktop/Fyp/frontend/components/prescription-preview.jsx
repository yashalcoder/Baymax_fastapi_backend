"use client"

export default function PrescriptionPreview({ data }) {
  if (!data) {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-sm text-muted-foreground">
        Fill the form and click “Generate Prescription” to preview here.
      </div>
    )
  }

  const { patientName, symptoms, medicines, dosage, duration, allergies, warnings, createdAt } = data

  const asText = () => {
    const lines = [
      "BayMax+ Prescription",
      `Date: ${new Date(createdAt).toLocaleString()}`,
      `Patient: ${patientName}`,
      "",
      "Symptoms:",
      symptoms || "-",
      "",
      "Medicines:",
      medicines || "-",
      "",
      "Dosage:",
      dosage || "-",
      "",
      `Duration: ${duration || "-"}`,
      `Known Allergies: ${allergies || "-"}`,
      "",
      ...(warnings && warnings.length ? ["Warnings:", ...warnings] : []),
      "",
      "— Generated via BayMax+ (demo export)",
    ]
    return lines.join("\n")
  }

  function downloadTxt() {
    const blob = new Blob([asText()], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `Prescription_${patientName || "patient"}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  function printPdf() {
    window.print()
  }

  function sendToPatient() {
    alert("This is a demo. Integrate your messaging/email system to send the prescription to the patient.")
  }

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-foreground">Prescription Preview</h3>
          <p className="text-xs text-muted-foreground">Generated on {new Date(createdAt).toLocaleString()}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={printPdf}
            className="rounded-md bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-primary"
          >
            Print / Save PDF
          </button>
          <button
            onClick={downloadTxt}
            className="rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-primary"
          >
            Download .txt
          </button>
          <button
            onClick={sendToPatient}
            className="rounded-md border border-border bg-background px-3 py-1.5 text-xs font-medium hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-primary"
          >
            Send to Patient
          </button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-md border border-border bg-background p-4">
          <p className="text-sm text-muted-foreground">Patient</p>
          <p className="text-lg font-semibold">{patientName || "—"}</p>
          <div className="mt-3 grid gap-2 text-sm">
            <div>
              <p className="font-medium">Symptoms</p>
              <p className="text-muted-foreground whitespace-pre-wrap">{symptoms || "—"}</p>
            </div>
            <div>
              <p className="font-medium">Duration</p>
              <p className="text-muted-foreground">{duration || "—"}</p>
            </div>
            <div>
              <p className="font-medium">Known Allergies</p>
              <p className="text-muted-foreground">{allergies || "—"}</p>
            </div>
          </div>
        </div>

        <div className="rounded-md border border-border bg-background p-4">
          <p className="text-sm text-muted-foreground">Medication Plan</p>
          <div className="mt-2 grid gap-2">
            <div>
              <p className="font-medium">Medicines</p>
              <pre className="text-sm whitespace-pre-wrap">{medicines || "—"}</pre>
            </div>
            <div>
              <p className="font-medium">Dosage</p>
              <pre className="text-sm whitespace-pre-wrap">{dosage || "—"}</pre>
            </div>
          </div>
        </div>
      </div>

      {warnings && warnings.length > 0 && (
        <div className="mt-4 rounded-md border border-border bg-background p-4">
          <p className="text-sm font-medium text-foreground">Warnings</p>
          <ul className="mt-1 list-disc pl-5 text-sm">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-4 text-xs text-muted-foreground">
        Note: This is a generated preview. Clinician must review for accuracy before finalizing.
      </p>
    </div>
  )
}
