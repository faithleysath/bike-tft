import type {
  AlgorithmKey,
  DecisionPayload,
  Meta,
  ModelKey,
  RunSummary,
  StationDetail,
  TimelineItem
} from "./types";

async function request<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export function getMeta(): Promise<Meta> {
  return request<Meta>("/api/meta");
}

export function getTimeline(): Promise<TimelineItem[]> {
  return request<TimelineItem[]>("/api/timeline");
}

export function getRunSummary(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/api/runs/summary");
}

export function getDecision(params: {
  ts: string;
  model: ModelKey;
  algorithm: AlgorithmKey;
  cap: number;
}): Promise<DecisionPayload> {
  const query = new URLSearchParams({
    ts: params.ts,
    model: params.model,
    algorithm: params.algorithm,
    cap: String(params.cap)
  });
  return request<DecisionPayload>(`/api/decision?${query.toString()}`);
}

export function getStationDetail(params: {
  nodeIdx: number;
  ts: string;
  model: ModelKey;
  algorithm: AlgorithmKey;
  cap: number;
}): Promise<StationDetail> {
  const query = new URLSearchParams({
    ts: params.ts,
    model: params.model,
    algorithm: params.algorithm,
    cap: String(params.cap)
  });
  return request<StationDetail>(`/api/station/${params.nodeIdx}?${query.toString()}`);
}

export function toInputDateTime(value: string): string {
  return value.replace(" ", "T").slice(0, 16);
}

export function toApiDateTime(value: string): string {
  if (!value) {
    return value;
  }
  return `${value.replace("T", " ")}:00`;
}
