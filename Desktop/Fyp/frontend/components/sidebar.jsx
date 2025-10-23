"use client";

import { useState } from "react";
import {
  Stethoscope,
  Activity,
  Users,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  const menuItems = [
    { icon: Activity, label: "Dashboard", active: true },
    { icon: Users, label: "Patients" },
    { icon: Stethoscope, label: "Diagnosis" },
    { icon: FileText, label: "Prescriptions" },
    { icon: Settings, label: "Settings" },
  ];

  return (
    <motion.aside
      animate={{ width: collapsed ? 80 : 260 }}
      transition={{ type: "spring", stiffness: 120, damping: 15 }}
      className="fixed left-0 top-0 h-screen bg-card border-r border-border shadow-lg z-20 flex flex-col"
    >
      {/* Header / Logo Section */}
      <div className="p-4 flex items-center justify-between border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
            <Stethoscope className="w-5 h-5 text-white" />
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
              >
                <h2 className="font-bold text-lg text-foreground">Baymax+</h2>
                <p className="text-xs text-muted-foreground">Doctor Portal</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 rounded-lg hover:bg-muted transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Nav Items */}
      <nav className="p-4 space-y-2 flex-1">
        {menuItems.map((item, idx) => (
          <button
            key={idx}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
              item.active
                ? "bg-gradient-to-r from-blue-600 to-cyan-500 text-white shadow"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            }`}
          >
            <item.icon className="w-5 h-5 shrink-0" />
            <AnimatePresence>
              {!collapsed && (
                <motion.span
                  initial={{ opacity: 0, x: -5 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -5 }}
                  className="font-medium"
                >
                  {item.label}
                </motion.span>
              )}
            </AnimatePresence>
          </button>
        ))}
      </nav>

      {/* Footer (Optional Settings or Collapse Info) */}
      <div className="p-4 border-t border-border text-xs text-muted-foreground">
        <AnimatePresence>
          {!collapsed && <motion.p animate={{ opacity: 1 }}>v1.0.0</motion.p>}
        </AnimatePresence>
      </div>
    </motion.aside>
  );
}
