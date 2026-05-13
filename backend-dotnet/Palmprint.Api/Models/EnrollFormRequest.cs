using Microsoft.AspNetCore.Http;

namespace Palmprint.Api.Models;

public class EnrollFormRequest
{
    public Guid TenantId { get; set; }

    public string FullName { get; set; } = string.Empty;

    public string ExternalId { get; set; } = string.Empty;

    public IFormFile? Image { get; set; }
}