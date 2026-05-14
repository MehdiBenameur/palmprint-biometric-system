using Microsoft.AspNetCore.Http;

namespace Palmprint.Api.Models;

public class IdentifyFormRequest
{
    public Guid TenantId { get; set; }

    public IFormFile? Image { get; set; }
}