import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const schedule = [
  { time: "09:00 AM", name: "Ahmed Ali", type: "Follow-up • 30 min" },
  { time: "10:00 AM", name: "Fatima Khan", type: "Consultation • 45 min" },
  { time: "11:30 AM", name: "Hassan Raza", type: "Check-up • 30 min" },
  { time: "02:00 PM", name: "Ayesha Malik", type: "Review • 20 min" },
];

export default function TodaySchedule() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Today&apos;s Schedule</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {schedule.map((s) => (
          <div
            key={s.time}
            className="flex items-center justify-between rounded-md border px-3 py-2"
          >
            <div className="leading-tight">
              <div className="text-xs text-muted-foreground">Time</div>
              <div className="font-medium">{s.time}</div>
            </div>
            <div className="flex-1 px-4">
              <div className="font-medium">{s.name}</div>
              <div className="text-xs text-muted-foreground">{s.type}</div>
            </div>
            <button className="text-sm text-blue-600 hover:underline">
              Start
            </button>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
