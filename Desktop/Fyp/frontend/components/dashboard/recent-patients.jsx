import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import Link from "next/link";

const patients = [
  { initials: "AA", name: "Ahmed Ali", age: 45, condition: "Hypertension" },
  { initials: "FK", name: "Fatima Khan", age: 32, condition: "Diabetes" },
  { initials: "HR", name: "Hassan Raza", age: 58, condition: "Cardiac" },
  { initials: "AM", name: "Ayesha Malik", age: 27, condition: "Asthma" },
];

export default function RecentPatients() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Patients</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {patients.map((p) => (
          <div
            key={p.name}
            className="flex items-center justify-between rounded-md border px-3 py-2"
          >
            <div className="flex items-center gap-3">
              <div className="size-9 rounded-full bg-muted flex items-center justify-center text-sm font-medium">
                {p.initials}
              </div>
              <div className="leading-tight">
                <div className="font-medium">{p.name}</div>
                <div className="text-xs text-muted-foreground">
                  {p.age} years • {p.condition}
                </div>
              </div>
            </div>
            <Link
              href={`/patients/${encodeURIComponent(p.name)}`}
              className="text-sm text-blue-600 hover:underline"
            >
              View
            </Link>
          </div>
        ))}
        <Link
          href="/patients"
          className="text-sm text-center text-blue-600 hover:underline mt-1"
        >
          View All Patients
        </Link>
      </CardContent>
    </Card>
  );
}
