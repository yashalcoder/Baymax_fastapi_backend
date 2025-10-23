"use client";

import { SidebarTrigger } from "@/components/ui/sidebar";

export default function TopNav({ children }) {
  return (
    <div className="sticky top-0 z-10 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
      <div className="flex items-center justify-between gap-4 px-4 py-2">
        <div className="flex items-center gap-2">
          <SidebarTrigger />
          <span className="sr-only">Toggle sidebar</span>
        </div>
        <div className="flex-1 flex items-center justify-end gap-3">
          {children}
        </div>
      </div>
    </div>
  );
}
