export interface DetectionItem {
  label: string;
  confidence: number;
  box: [number, number, number, number];
}

export interface BreakdownItem {
  type: string;
  count: number;
  percentage: number;
}

export interface InspectionResult {
  success: boolean;
  latency_ms: number;
  total_count: number;
  unique_types: number;
  occupancy_rate: number;
  total_slots: number;
  breakdown: BreakdownItem[];
  detections: DetectionItem[];
  annotated_image: string; // Base64 data URL
}
