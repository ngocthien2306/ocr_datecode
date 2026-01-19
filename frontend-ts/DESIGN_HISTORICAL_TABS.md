# Design: Historical Page with 2 Tabs

## Current State Analysis

### Backend API Available
**Endpoint:** `/api/inference-results/`
- **Methods:**
  - GET `/` - List results with filters
  - GET `/count` - Get count
  - GET `/{id}` - Get by ID
  - DELETE `/{id}` - Delete result

**Filters:**
- `skip`, `limit` - Pagination
- `recipe_id` - Filter by recipe
- `pass_fail` - Filter by PASS/FAIL
- `start_date`, `end_date` - Date range filter

**Response Model:**
```typescript
interface InferenceResultResponse {
  id: string;
  recipe_id: string;
  recipe_name: string;
  product_pass_fail: 'PASS' | 'FAIL';
  camera_results: CameraResult[];
  metadata: {
    total_cameras: number;
    total_frames: number;
    inference_stats: {
      avg_confidence: number;
      total_inliers: number;
      total_matches: number;
      per_camera_stats: PerCameraStats[];
    };
  };
  timestamp: string;
  created_at: string;
}
```

### Current Historical.tsx
- Hardcoded data for 7 days
- Production Trends chart (canvas)
- Daily Production Records table
- Mock data: `{ date, camera1, camera2, camera3 }`

---

## Proposed Design: 2 Tabs

### Tab Structure

```
┌─────────────────────────────────────────────────────────┐
│ Historical Data                     [Export Data]      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Inspection Results] [Production Analytics]           │ <- Tabs
│  ═══════════════════                                    │
│                                                         │
│  [Tab Content Area]                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Tab 1: Inspection Results (Chi tiết từng inference)

### Purpose
- View detailed inspection results per product
- Filter by date range, recipe, PASS/FAIL
- View text verification details
- Export results to CSV/Excel

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ 🔍 Inspection Results                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🎛️ Filters:                                                    │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ Date Range: [Last 7 Days ▼] Recipe: [All ▼] Status: [All ▼]││
│ │ 📅 From: [2026-01-02] To: [2026-01-09]     [Apply Filter]  ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ 📊 Summary:                                                    │
│ ┌────────────┬────────────┬────────────┬────────────┐        │
│ │ Total: 345 │ PASS: 320  │ FAIL: 25   │ Rate: 92.8%│        │
│ └────────────┴────────────┴────────────┴────────────┘        │
│                                                                 │
│ 📋 Results Table:                                              │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ Time       Recipe      Result  Cameras Text Verify Actions││
│ │─────────────────────────────────────────────────────────────│
│ │ 14:32:45  Pepper 400g  PASS    4/4     2/2 ✓      [View] ││ <- Click to expand
│ │ 14:31:20  Pepper 400g  FAIL    4/4     1/2 ✗      [View] ││
│ │   └─ Camera CAM001: Text mismatch - Expected "LOT:123"    ││ <- Expanded detail
│ │      Got "LOT:I23" (Confidence: 45%)                       ││
│ │ 14:30:05  Pepper 400g  PASS    4/4     2/2 ✓      [View] ││
│ │ 14:29:10  Salt 500g    PASS    3/3     N/A        [View] ││
│ │ ...                                                        ││
│ └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Showing 1-50 of 345       [< 1 2 3 4 ... 7 >]    [50 ▼]      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Table Columns

| Column | Description | Width | Sortable |
|--------|-------------|-------|----------|
| **Timestamp** | Inspection time | 100px | ✓ |
| **Recipe** | Recipe name + product code | 180px | ✓ |
| **Result** | PASS/FAIL badge | 80px | ✓ |
| **Cameras** | Success/Total (e.g., "4/4") | 80px | - |
| **Text Verification** | Match status (e.g., "2/2 ✓") | 120px | ✓ |
| **Confidence** | Avg confidence % | 90px | ✓ |
| **Actions** | View/Export/Delete | 100px | - |

### Expandable Row Detail

When clicking a row, expand to show:
```
┌─────────────────────────────────────────────────────────┐
│ 🔍 Inspection Details - ID: abc123                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📷 Camera Results:                                     │
│ ┌─────────────────────────────────────────────────────┐│
│ │ CAM001 (PASS) - Conf: 98%                          ││
│ │   ├─ Inliers: 2,450 / 3,200                        ││
│ │   ├─ Processing: 421ms (TRT: 264ms)                ││
│ │   └─ Text Verification:                             ││
│ │       • Region 1: "Table Grind" ✓ (98%)            ││
│ │       • Region 2: "Black Pepper" ✗ (0%)            ││
│ │         Expected: "Black Pepper"                    ││
│ │         Got: "cYanmaGlow"                           ││
│ │                                                      ││
│ │ CAM002 (PASS) - Conf: 95%                          ││
│ │   └─ No text verification                           ││
│ └─────────────────────────────────────────────────────┘│
│                                                         │
│ 📸 Images:                                             │
│ ┌──────────┬──────────┬──────────┬──────────┐        │
│ │ [IMG1]   │ [IMG2]   │ [IMG3]   │ [IMG4]   │        │
│ │ CAM001   │ CAM002   │ CAM003   │ CAM004   │        │
│ └──────────┴──────────┴──────────┴──────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Features

#### 1. **Filters**
- Date range picker (preset + custom)
- Recipe dropdown (fetch from API)
- Status filter (All / PASS / FAIL)
- Apply/Reset buttons

#### 2. **Sorting**
- Click column headers to sort
- Indicator arrows (▲/▼)
- Default: Sort by timestamp DESC (newest first)

#### 3. **Pagination**
- Server-side pagination (skip/limit)
- Page size selector: 10, 25, 50, 100
- First/Prev/Next/Last buttons
- Jump to page input

#### 4. **Text Verification Highlight**
- Green "2/2 ✓" if all match
- Red "1/2 ✗" if any mismatch
- Gray "N/A" if no verification

#### 5. **Actions**
- **View** - Expand row / Modal with details
- **Export** - Download single result as JSON/CSV
- **Delete** - Delete result (admin only)

#### 6. **Bulk Actions**
- Checkbox to select multiple rows
- Bulk export selected
- Bulk delete selected (admin only)

---

## Tab 2: Production Analytics (Tổng hợp thống kê)

### Purpose
- View production trends over time
- Aggregate stats by date/recipe/camera
- Production charts (will implement later)
- Daily production table

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ 📊 Production Analytics                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📈 Summary Cards:                                               │
│ ┌────────────┬────────────┬────────────┬────────────┐        │
│ │ Total      │ Avg Daily  │ Best       │ Pass       │        │
│ │ Products   │ Output     │ Camera     │ Rate       │        │
│ │ 8,687      │ 1,241      │ CAM001     │ 92.8%      │        │
│ │ +15% ↑     │ Per day    │ 3,189      │ +2.3% ↑    │        │
│ └────────────┴────────────┴────────────┴────────────┘        │
│                                                                 │
│ 🎛️ Controls:                                                   │
│ Date Range: [Last 7 Days ▼]  Camera: [All Cameras ▼]         │
│                                                                 │
│ 📊 Production Trends Chart:                                    │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │                        [Chart Area]                         ││
│ │  500 ┤                              ●                       ││
│ │  400 ┤         ●──●──●──●──●──●                            ││
│ │  300 ┤    ●                                                 ││
│ │  200 ┤                                                      ││
│ │  100 ┤                                                      ││
│ │    0 ┴─────┬─────┬─────┬─────┬─────┬─────┬─────           ││
│ │         1/2  1/3  1/4  1/5  1/6  1/7  1/8                  ││
│ │                                                              ││
│ │  Legend: ■ CAM001  ■ CAM002  ■ CAM003                      ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ 📋 Daily Production Records:                                   │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ Date       CAM001  CAM002  CAM003  Total  Pass  Fail Trend││
│ │─────────────────────────────────────────────────────────────│
│ │ 2026-01-09  445    412    355    1,212   1,127   85   ↑12 ││
│ │ 2026-01-08  438    420    362    1,220   1,142   78   ↑8  ││
│ │ 2026-01-07  452    415    358    1,225   1,138   87   ↓5  ││
│ │ ...                                                        ││
│ └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Aggregation

Query API to aggregate by date:
```typescript
interface DailyStats {
  date: string;
  total_count: number;
  pass_count: number;
  fail_count: number;
  pass_rate: number;
  per_camera: {
    [camera_serial]: {
      count: number;
      pass: number;
      fail: number;
    }
  };
}
```

**Backend API needed** (new endpoint):
```
GET /api/inference-results/stats/daily
  ?start_date=2026-01-02
  &end_date=2026-01-09
  &recipe_id=xxx (optional)
```

Response:
```json
{
  "summary": {
    "total_products": 8687,
    "total_pass": 8054,
    "total_fail": 633,
    "pass_rate": 92.71,
    "avg_daily_output": 1241,
    "best_camera": {
      "serial": "CAM001",
      "count": 3189
    }
  },
  "daily_stats": [
    {
      "date": "2026-01-09",
      "total": 1212,
      "pass": 1127,
      "fail": 85,
      "cameras": {
        "CAM001": { "total": 445, "pass": 420, "fail": 25 },
        "CAM002": { "total": 412, "pass": 395, "fail": 17 },
        "CAM003": { "total": 355, "pass": 312, "fail": 43 }
      }
    }
  ]
}
```

### Features

#### 1. **Summary Cards**
- Total products in period
- Average daily output
- Best performing camera
- Pass rate %
- Trend vs previous period

#### 2. **Chart (TODO later)**
- Line chart for trends
- Multiple series (per camera)
- Date range zoom
- Interactive tooltips

#### 3. **Daily Table**
- Date column (sortable)
- Per-camera counts
- Total/Pass/Fail
- Trend indicator
- Export to CSV/Excel

---

## Component Structure

```
Historical.tsx (Main)
├── Tab Navigation (State: activeTab)
├── Tab1: InspectionResultsTab
│   ├── FilterBar
│   │   ├── DateRangePicker
│   │   ├── RecipeSelect
│   │   └── StatusSelect
│   ├── SummaryStats
│   ├── ResultsTable
│   │   ├── TableHeader (sortable)
│   │   ├── TableRow (expandable)
│   │   │   └── ExpandedDetail
│   │   │       ├── CameraResults
│   │   │       ├── TextVerification
│   │   │       └── ImageGallery
│   │   └── TablePagination
│   └── BulkActions
└── Tab2: ProductionAnalyticsTab
    ├── SummaryCards
    ├── ChartControls
    ├── ProductionTrendsChart (TODO)
    └── DailyProductionTable
```

---

## API Service Layer

Create: `frontend-ts/src/services/inferenceResults.ts`

```typescript
export const inferenceResultsAPI = {
  // Get results with filters
  getResults: async (params: {
    skip?: number;
    limit?: number;
    recipe_id?: string;
    pass_fail?: 'PASS' | 'FAIL';
    start_date?: string;
    end_date?: string;
  }) => {
    const response = await api.get('/inference-results/', { params });
    return response.data;
  },

  // Get count
  getCount: async (filters: any) => {
    const response = await api.get('/inference-results/count', { params: filters });
    return response.data;
  },

  // Get by ID
  getById: async (id: string) => {
    const response = await api.get(`/inference-results/${id}`);
    return response.data;
  },

  // Delete
  delete: async (id: string) => {
    const response = await api.delete(`/inference-results/${id}`);
    return response.data;
  },

  // Get daily stats (TODO: backend endpoint)
  getDailyStats: async (params: {
    start_date: string;
    end_date: string;
    recipe_id?: string;
  }) => {
    const response = await api.get('/inference-results/stats/daily', { params });
    return response.data;
  }
};
```

---

## State Management

```typescript
// Tab 1 State
const [activeTab, setActiveTab] = useState<'results' | 'analytics'>('results');

// Filters
const [dateRange, setDateRange] = useState('7days');
const [customDateRange, setCustomDateRange] = useState({ from: '', to: '' });
const [selectedRecipe, setSelectedRecipe] = useState<string | null>(null);
const [statusFilter, setStatusFilter] = useState<'all' | 'PASS' | 'FAIL'>('all');

// Pagination
const [currentPage, setCurrentPage] = useState(1);
const [pageSize, setPageSize] = useState(50);
const [totalCount, setTotalCount] = useState(0);

// Data
const [results, setResults] = useState<InferenceResultResponse[]>([]);
const [loading, setLoading] = useState(false);
const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

// Selection
const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
```

---

## Responsive Design

### Desktop (> 1024px)
- Full table with all columns
- Side-by-side layout for filters
- Large images in expanded view

### Tablet (768px - 1024px)
- Compact table (hide some columns)
- Stack filters vertically
- Smaller images

### Mobile (< 768px)
- Card-based layout instead of table
- Each result as a card:
  ```
  ┌───────────────────────┐
  │ 14:32:45  PASS       │
  │ Pepper 400g           │
  │ Cameras: 4/4          │
  │ Text: 2/2 ✓          │
  │ [View Details]        │
  └───────────────────────┘
  ```
- Filters in bottom sheet / drawer

---

## Implementation Priority

### Phase 1: Core Functionality
1. Tab navigation structure
2. API service integration
3. Tab 1: Results table with basic columns
4. Pagination
5. Date range filter

### Phase 2: Advanced Features
6. Expandable row details
7. Text verification display
8. Image gallery
9. Sorting
10. Status filter

### Phase 3: Tab 2
11. Daily stats API endpoint (backend)
12. Summary cards with real data
13. Daily production table
14. Chart (later)

### Phase 4: Polish
15. Export to CSV/Excel
16. Bulk actions
17. Responsive design
18. Loading states / skeleton
19. Error handling

---

## UX Considerations

1. **Default View**: Tab 1 (Inspection Results) - most frequently used
2. **Auto-refresh**: Optional auto-refresh for Tab 1 (newest results)
3. **Persistent Filters**: Save filter state to localStorage
4. **Quick Filters**: Preset buttons (Today, Yesterday, This Week)
5. **Search**: Add search by product code / recipe name
6. **Performance**: Virtual scrolling for large datasets (100+ rows)
7. **Export**: Include text verification in CSV export

---

## Design Rationale

### Why 2 Tabs?

**Tab 1 (Detailed):**
- For QC operators to investigate FAIL results
- Debug text verification issues
- View specific inspection details
- Export for reporting

**Tab 2 (Aggregated):**
- For managers/supervisors
- Overview of production performance
- Trend analysis
- Daily/weekly reporting

### Why Not Combine?

- Different use cases
- Different data granularity
- Different audiences
- Cleaner UI (less cluttered)

### Alternative Considered: Single Page with Sections

Pros:
- All data visible at once
- No tab switching

Cons:
- Too much scrolling
- Overwhelming for new users
- Harder to focus on specific task

**Decision: 2 Tabs** ✓
