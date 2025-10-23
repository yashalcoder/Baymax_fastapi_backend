"use client"

export default function PatientList({ patients = defaultPatients }) {
  return (
    <div className="card">
      <div className="px-4 py-3 border-b border-[color:var(--color-border)] flex items-center justify-between">
        <h3 className="font-semibold">Patients</h3>
        <span className="text-xs text-muted-foreground">{patients.length} total</span>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left text-muted-foreground">
              <th className="px-4 py-3 font-medium">Name</th>
              <th className="px-4 py-3 font-medium">Age</th>
              <th className="px-4 py-3 font-medium">Last Visit</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p, idx) => (
              <tr
                key={idx} // Added key property
                className={[
                  "border-t border-[color:var(--color-border)]",
                  idx % 2 ? "bg-[color:var(--color-secondary)]" : "",
                ].join(" ")}
              >
                <td className="px-4 py-3">{p.name}</td>
                <td className="px-4 py-3">{p.age}</td>
                <td className="px-4 py-3">{p.lastVisit}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const defaultPatients = [
  { name: "Ahmed Khan", age: 45, lastVisit: "2025-10-02" },
  { name: "Sara Ali", age: 32, lastVisit: "2025-09-28" },
  { name: "Bilal Hussain", age: 54, lastVisit: "2025-09-15" },
  { name: "Fatima Noor", age: 29, lastVisit: "2025-10-10" },
  { name: "Usman Raza", age: 38, lastVisit: "2025-10-08" },
]
