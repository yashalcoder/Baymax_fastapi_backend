"use client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Activity,
  Calendar,
  FileText,
  Settings,
  Stethoscope,
  Users,
  Plus,
  ChevronRight,
} from "lucide-react";
import Sidebar from "@/components/sidebar";
import { useRouter } from "next/navigation";
const DoctorDashboard = () => {
  const router = useRouter();

  const patients = [
    {
      id: 1,
      name: "Ahmad Khan",
      age: 45,
      lastVisit: "2025-01-10",
      condition: "Diabetes Follow-up",
    },
    {
      id: 2,
      name: "Fatima Ali",
      age: 32,
      lastVisit: "2025-01-10",
      condition: "Hypertension",
    },
    {
      id: 3,
      name: "Hassan Raza",
      age: 28,
      lastVisit: "2025-01-09",
      condition: "Annual Checkup",
    },
    {
      id: 4,
      name: "Ayesha Malik",
      age: 51,
      lastVisit: "2025-01-08",
      condition: "Arthritis",
    },
  ];

  const stats = [
    {
      label: "Total Patients",
      value: "247",
      icon: Users,
      color: "text-primary",
    },
    {
      label: "Today's Consultations",
      value: "12",
      icon: Activity,
      color: "text-medical-info",
    },
    {
      label: "Pending Reports",
      value: "8",
      icon: FileText,
      color: "text-medical-warning",
    },
    {
      label: "Upcoming Appointments",
      value: "15",
      icon: Calendar,
      color: "text-medical-success",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-secondary/30 to-accent/20">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="ml-64 p-8">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold text-foreground mb-2">
                Welcome, Dr. Ahmed
              </h1>
              <p className="text-muted-foreground">
                Here's what's happening with your patients today
              </p>
            </div>
            <Button
              className="bg-hero-gradient text-white shadow-medical-lg hover:opacity-90 transition-opacity"
              onClick={() => router.push("/doctor/transcription")}
            >
              <Plus className="w-4 h-4 mr-2" />
              New Consultation
            </Button>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, idx) => (
              <Card
                key={idx}
                className="bg-gradient-card border-border shadow-medical-md hover:shadow-medical-lg transition-shadow"
              >
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground font-medium">
                        {stat.label}
                      </p>
                      <p className="text-3xl font-bold text-foreground mt-2">
                        {stat.value}
                      </p>
                    </div>
                    <div
                      className={`w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center ${stat.color}`}
                    >
                      <stat.icon className="w-6 h-6" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Patients List */}
          <Card className="shadow-medical-lg border-border">
            <CardHeader className="border-b border-border">
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="text-2xl">Recent Patients</CardTitle>
                  <CardDescription className="mt-1">
                    View and manage your patient consultations
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  className="border-primary text-primary hover:bg-primary/10"
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Add New Patient
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border">
                {patients.map((patient) => (
                  <div
                    key={patient.id}
                    className="p-6 hover:bg-secondary/50 transition-colors cursor-pointer flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-full bg-hero-gradient flex items-center justify-center text-white font-semibold shadow-medical-md">
                        {patient.name
                          .split(" ")
                          .map((n) => n[0])
                          .join("")}
                      </div>
                      <div>
                        <h3 className="font-semibold text-foreground">
                          {patient.name}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          Age: {patient.age} • Last visit: {patient.lastVisit}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="px-3 py-1 rounded-full bg-accent text-accent-foreground text-sm font-medium">
                        {patient.condition}
                      </span>
                      <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default DoctorDashboard;
